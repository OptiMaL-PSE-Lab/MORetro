import logging
import torch
import torch.nn as nn
from abc import abstractmethod
from typing import Any
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

from rdchiral.main import rdchiralRun
from rdchiral.initialization import rdchiralReactants, rdchiralReaction

logger = logging.getLogger(__name__)


def get_activation(name: str) -> nn.Module:
    _activations = {
        "relu": nn.ReLU(),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
        "leakyrelu": nn.LeakyReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
    }

    return _activations[name]


class Dense(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_act: nn.Module):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.hidden_act = hidden_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hidden_act(self.linear(x))


class TemplateModel(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def _run_templates(self):
        pass

    def _smis_to_fp(self, smiles: str | list[str], fp_size: int = 2048) -> torch.Tensor:
        """
        Convert a list of SMILES string to a fingerprint tensor.
        """
        fps = []
        if isinstance(smiles, str):
            smiles = [smiles]

        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            fp = GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=fp_size, useChirality=True
            )
            fp = torch.tensor(fp, dtype=torch.float)
            fps.append(fp)

        if len(fps) == 1:
            return fps[0]
        return torch.stack(fps)


class TemplRel(TemplateModel):
    def __init__(self, args):
        super().__init__()
        if isinstance(args.hidden_sizes, str):
            self.hidden_sizes = [int(size) for size in args.hidden_sizes.split(",")]

        self.args = args
        self.layers = self._build_layers(args)
        self.output_layer = nn.Linear(
            self.hidden_sizes[-1], args.n_templates, bias=True
        )

        self.dropout = nn.Dropout(args.dropout)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")

    def _build_layers(self, args) -> nn.ModuleList:
        hidden_act = get_activation(args.hidden_activation)
        # input projection layer; no skip connection here
        layers = nn.ModuleList(
            [Dense(args.fp_size, self.hidden_sizes[0], hidden_act=hidden_act)]
        )

        for layer_i in range(len(self.hidden_sizes) - 1):
            in_features = self.hidden_sizes[layer_i]
            out_features = self.hidden_sizes[layer_i + 1]

            if args.skip_connection == "none":
                layer = Dense(in_features, out_features, hidden_act=hidden_act)
            else:
                raise ValueError(f"Unsupported skip_connection: {args.skip_connection}")

            layers.append(layer)

        return layers

    def predict(
        self, products: str | list[str], top_n: int, templates: dict
    ) -> list[list[dict[str, Any]]]:
        # Handle both single product and list of products
        if isinstance(products, str):
            products = [products]
            single_input = True
        else:
            single_input = False

        target_fp = self._smis_to_fp(products)

        # Get fingerprints and rdchiral reactants for each product
        target_rds = []
        for prod in products:
            target_rds.append(rdchiralReactants(prod))

        with torch.no_grad():
            # Explicitly set model to eval mode to ensure no dropout
            self.eval()
            output = self(target_fp)

        # Process all products with the same logic
        all_predictions = []
        probs = torch.softmax(output, dim=1 if len(products) > 1 else 0)
        
        # Handle dimension for single vs multiple products
        if single_input:
            probs = probs.unsqueeze(0)  # Add batch dimension for consistency

        for idx, (prod, target_rd) in enumerate(zip(products, target_rds)):
            top_scores, top_indices = torch.topk(probs[idx], top_n)
            top_scores = top_scores.detach().numpy()
            top_indices = top_indices.detach().numpy()

            predictions = []
            for i in range(top_n):
                template = templates[top_indices[i]]
                pred_reactants = self._run_templates(target_rd, template)
                if len(pred_reactants) > 0:
                    for output_react in pred_reactants:
                        predictions.append(
                            {
                                "score": top_scores[i],
                                "reactants": output_react,
                                "template": template,
                            }
                        )

            predictions = self._postprocessing(predictions, prod)
            all_predictions.append(predictions)

        return all_predictions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)
        logits = self.output_layer(x)  # returning *unnormalized* logits
        return logits

    def _run_templates(self, product: rdchiralReactants, template: str) -> list[str]:
        """
        Run the template given product and corresponding template

        Args:
            product: The product to run the template on.
            template: The template to use for the product.

        Returns:
            The generated output after applying the template to the product.

        """
        reactants = template.split(">>")[0].split(".")
        if len(reactants) > 1:
            template = "(" + template.replace(">>", ")>>")
        template_rd = rdchiralReaction(template)
        try:
            output = rdchiralRun(template_rd, product)
        except Exception as e:
            logger.error(f"Error occurred while running template: {e}")
            return []
        result = []
        for out in output:
            result.append(out.split("."))
        return result

    def _postprocessing(self, predictions: list[dict], product: str) -> list[dict]:
        """
        Only retain unique reactants, templates and scores are added together
        """
        prec_to_score = {}
        prec_to_template = {}
        for i in range(len(predictions)):
            prec = frozenset(predictions[i]["reactants"])
            if prec in prec_to_score:
                prec_to_score[prec] += predictions[i]["score"]
                prec_to_template[prec].append(predictions[i]["template"])
            else:
                prec_to_score[prec] = predictions[i]["score"]
                prec_to_template[prec] = [predictions[i]["template"]]

        # Renormalize scores
        total_score = sum(prec_to_score.values())
        for prec in prec_to_score:
            prec_to_score[prec] /= total_score
        final_predictions = []
        for prec in sorted(prec_to_score.keys(), key=lambda x: sorted(list(x))):
            final_predictions.append(
                {
                    "rxn_smiles": ".".join(prec) + ">>" + product,
                    "score": prec_to_score[prec],
                    "template": prec_to_template[prec],
                    "reactants": list(prec),
                }
            )

        return final_predictions
