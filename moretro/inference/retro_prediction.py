import gin
import json
import torch
from logging import Logger
from typing import Any, Optional

from moretro.inference.template_models import TemplRel
from pathlib import Path

logger = Logger(__name__)
file_path = Path(__file__).parent

@gin.configurable()
class OneStepModel:
    """
    A one-step model class for retro prediction.
    This class incorporates different one step models
    """

    def __init__(self, model_type: str, checkpoint_path: str, template_path: Optional[str] = None):
        self.model_type = model_type 
        self.checkpoint_path = file_path.parent / checkpoint_path
        self.template_path = file_path.parent / template_path if template_path else None
        if self.template_path: 
                with open(self.template_path, "r") as f:
                    template_dict = json.load(f)
                self.templates = {}
                for k, v in template_dict.items():
                    self.templates[int(k)] = v
        else:
            logger.info("No template path provided, expected for non-template models.")

        if model_type == "st":
            retro_checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            pretrain_args = retro_checkpoint["args"]
            self.model = TemplRel(pretrain_args)
            state_dict = retro_checkpoint["state_dict"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            #* Add new models here
        self.model.eval()

    def predict(self, target: str | list[str], top_n: int=50) -> list[list[dict[str, Any]]]: # type: ignore
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
        list[dict[str, Any]]
            A list of dictionaries containing the predicted retro reactions.
        Entries of dict must be: ["rxn_smiles", "reactants", "template"] where "template" can be None
        """
        predictions = self.model.predict(target, top_n, self.templates)
        return predictions
        

class ConditionPrediction:
    """
    Prediction of reaction conditions given the reaction string
    """

    def __init__(self, model_path: str):
        self.model_path = model_path

    def predict(self, reaction_smiles: str):
        """
        Predict the reaction conditions for a given reaction SMILES.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")
