import json
from logging import Logger
from pathlib import Path
from typing import Any

import gin
import torch

from moretro.external.quarc.quarc_predictor import QuarcPredictor
from moretro.external.template_models import TemplRel
from moretro.inference.calculate_costs import COST_MAPPING, calculate_costs
from moretro.utils.typing_hints import Predictions

logger = Logger(__name__)
file_path = Path(__file__).parent


@gin.configurable()
class OneStepModel:
    """
    A one-step model class for retro prediction.
    This class incorporates different one step models
    """

    def __init__(
        self,
        model_type: str,
        checkpoint_path: str,
        cost_functions: list[str],
        template_path: str | None = None,
    ):
        self.model_type = model_type
        self.checkpoint_path = file_path.parent / checkpoint_path
        self.template_path = file_path.parent / template_path if template_path else None
        self.condition_model = ConditionModel(gin.REQUIRED)  # type: ignore
        logger.info(f"Loading Single-Step Model from {self.checkpoint_path}")

        self.cost_functions = []
        for cost_name in cost_functions:
            if cost_name in COST_MAPPING:
                self.cost_functions.append(COST_MAPPING[cost_name])
            else:
                logger.error(f"Unknown cost function: {cost_name}")
                raise ValueError("Please ensure that all cost functions are defined")

        if self.template_path:
            with open(self.template_path, encoding="utf-8") as f:
                template_dict = json.load(f)
            self.templates = {}
            for k, v in template_dict.items():
                self.templates[int(k)] = v
        else:
            logger.info("No template path provided, expected for non-template models.")

        if model_type == "st":
            retro_checkpoint = torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=False
            )
            pretrain_args = retro_checkpoint["args"]
            self.model = TemplRel(pretrain_args)
            state_dict = retro_checkpoint["state_dict"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            # * Add new models here
        self.model.eval()

    def predict(self, target: str | list[str], top_n: int = 50) -> Predictions:
        """
        Predict the retro reactions for a given molecule or list of molecules up to top_n reactions.

        Parameters
        ----------
        target : str | list[str]
            The SMILES representation of the target molecule or a list of SMILES strings.
        top_n : int
            The number of top predictions to return.

        Returns
        -------
        Predictions
            A list of lists of dictionaries containing the predicted retro reactions.
            Each prediction dict contains: ["rxn_smiles", "reactants", "template", "score", "costs", "reagents", "temperature"]
        """
        # Get predictions from the underlying model
        if isinstance(target, list) and len(target) == 1:
            target = target[0]
        predictions = self.model.predict(target, top_n, self.templates)
        updated_predictions = self._add_cost_and_condition(predictions)
        return updated_predictions

    def _add_cost_and_condition(self, predictions: Predictions) -> Predictions:
        # Add cost calculations and missing fields to each prediction
        updated_predictions = []
        for mol_predictions in predictions:
            rxn_smiles = [pred["rxn_smiles"] for pred in mol_predictions]
            if not rxn_smiles:
                updated_predictions.append(mol_predictions)
                continue
            conditions = self.condition_model.predict(rxn_smiles)
            expanded_mol_predictions = []
            for pred, topk_cond in zip(mol_predictions, conditions, strict=True):
                for cond in topk_cond:
                    # Create a copy of the prediction for each condition
                    pred_copy = pred.copy()
                    pred_copy["temperature"] = cond["temperature"]
                    pred_copy["reagents"] = ".".join(cond["reagents"])
                    if "agent_amounts" in cond:
                        pred_copy["agent_amounts"] = cond["agent_amounts"]
                    costs = calculate_costs(pred_copy, self.cost_functions)
                    pred_copy["costs"] = costs
                    expanded_mol_predictions.append(pred_copy)

            updated_predictions.append(expanded_mol_predictions)
        return updated_predictions


@gin.configurable()
class ConditionModel:
    """
    Prediction of reaction conditions given the reaction string
    """

    def __init__(
        self,
        model_type: str,
        config_path: str,
        device: str,
        top_k: int,
        beam_size: int,
    ):
        self.model_type = model_type
        self.config_path = file_path.parent / config_path
        self.device = device
        self.top_k = top_k
        self.beam_size = beam_size
        if self.model_type == "quarc":
            self.model = QuarcPredictor(
                config_path=self.config_path, device=self.device
            )
        elif self.model_type == "rct":
            # TODO implement this
            raise NotImplementedError("R-CT model not implemented yet.")
        else:
            raise ValueError(f"Unsupported condition model type: {self.model_type}")

    def predict(self, rxn_smiles: list[str]) -> list[list[dict[str, Any]]]:
        """
        Predict the reaction conditions for a given reaction SMILES.
        This method should be implemented by subclasses.
        """
        results = self.model.predict(rxn_smiles, self.top_k, self.beam_size)
        results = self._clean_up_prediction(results)
        return results

    def _clean_up_prediction(
        self, predictions: list[list[dict[str, Any]]]
    ) -> list[list[dict[str, Any]]]:
        """
        Clean up the predictions by removing duplicate entries.
        """
        cleaned_predictions = []
        for mol_preds in predictions:
            seen = set()
            unique_preds = []
            for pred in mol_preds:
                # NOTE: This ignores the reagent amount for now.
                pred_tuple = (pred["temperature"], tuple(pred["reagents"]))
                if pred_tuple not in seen:
                    seen.add(pred_tuple)
                    unique_preds.append(pred)
            cleaned_predictions.append(unique_preds)
        return cleaned_predictions
