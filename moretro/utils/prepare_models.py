import pickle
import pandas as pd
import logging
from moretro.inference.retro_prediction import OneStepModel, ConditionPrediction

logger = logging.getLogger(__name__)


def prepare_starting_mols(file_path: str) -> set[str]:
    """
    Load building blocks from file

    Parameters:
        file_path (str): Path to the file containing building blocks

    Returns:
        set[str]: Set of building blocks (SMILES
    """

    if file_path.endswith(".csv"):
        starting_mol = set(pd.read_csv(file_path)["smiles"].tolist())
    elif file_path.endswith(".pkl"):
        with open(file_path, "rb") as f:
            starting_mol = pickle.load(f)
    else:
        raise ValueError("Unsupported file format. Use .csv or .pkl")

    logger.info(f"Loaded {len(starting_mol)} building blocks from {file_path}")
    return starting_mol


def prepare_retro_model(template_path: str, model_path: str) -> OneStepModel:
    """
    Loads the single-step retro model from path

    Parameters:
        template_path (str): Path to the template model file
        model_path (str): Path to the trained model file

    Returns:
        TemplateModel: Instance of the retro prediction model
    """
    logging.info(f"Loading templates from {template_path}")
    logging.info(f"Loading retro model from {model_path}")
    one_step_model = OneStepModel(model_path, template_path)
    return one_step_model


def prepare_condition_model():
    pass
