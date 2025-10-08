import json
from logging import Logger
from pathlib import Path

import gin
import torch

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
        self.condition_model = ConditionPrediction(gin.REQUIRED)  # type: ignore
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
        predictions = self._add_cost_and_condition(predictions)
        return predictions

    def _add_cost_and_condition(self, predictions: Predictions) -> Predictions:
        # Add cost calculations and missing fields to each prediction
        for mol_predictions in predictions:
            for pred in mol_predictions:
                costs = calculate_costs(pred, self.cost_functions)
                temp, reagents = self.condition_model.predict(pred["rxn_smiles"])
                pred["costs"] = costs
                pred["temperature"] = temp
                pred["reagents"] = reagents
        return predictions


@gin.configurable()
class ConditionPrediction:
    """
    Prediction of reaction conditions given the reaction string
    """

    def __init__(self, model_path: str):
        self.model_path = model_path

    def predict(self, rxn_smiles: str) -> tuple[int, str]:
        """
        Predict the reaction conditions for a given reaction SMILES.
        This method should be implemented by subclasses.
        """
        # TODO do this properly, for now dummy variables
        return 8, "int"
