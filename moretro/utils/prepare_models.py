import json
import pickle
import gin
import pandas as pd
import logging
from pathlib import Path
from typing import Callable

from moretro.inference.retro_prediction import ConditionPrediction
from moretro.inference.heuristic_functions import COST_MAPPING

logger = logging.getLogger(__name__)

@gin.configurable()
def prepare_starting_mols(file_path: str | Path) -> set[str]:
    """
    Load building blocks from file

    Parameters:
        file_path (str): Path to the file containing building blocks

    Returns:
        set[str]: Set of building blocks (SMILES
    """
    dir_path = Path(__file__).parent.parent
    file_path = dir_path / Path(file_path)
    if file_path.suffix == ".csv":
        starting_mol = set(pd.read_csv(file_path)["smiles"].tolist())
    elif file_path.suffix == ".pkl":
        with open(file_path, "rb") as f:
            starting_mol = pickle.load(f)
    elif file_path.suffix == ".json":
        with open(file_path, "r") as f:
            starting_mol = set(json.load(f))
    else:
        raise ValueError("Unsupported file format. Use .csv or .pkl or .json")

    logger.info(f"Loaded {len(starting_mol)} building blocks from {file_path}")
    return starting_mol

@gin.configurable()
def prepare_heuristic_fns(heuristics: list[str]) -> list[Callable[[str], float]]:
    heuristic_fs = []
    for heuristic in heuristics:
        if heuristic in COST_MAPPING:
            heuristic_fs.append(COST_MAPPING[heuristic])
        else:
            logger.error(f"Unknown heuristic: {heuristic}")
            raise ValueError("Please ensure that heuristic is defined for all cost functions")
    return heuristic_fs


def prepare_condition_model():
    pass
