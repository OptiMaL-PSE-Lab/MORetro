import warnings
from pathlib import Path

import numpy as np
import torch
from joblib import load
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.exceptions import InconsistentVersionWarning  # type: ignore

from moretro.external.molprice import MolPrice
from moretro.external.sa_score import sascorer_optimized
from moretro.external.value_fn import load_value_model

warnings.filterwarnings(action="ignore", category=InconsistentVersionWarning)
RDLogger.DisableLog("rdApp.*")  # type: ignore

# Instantiate the model once at module level to avoid overhead
model_path = Path(__file__).parent.parent / "models"
_price_model = MolPrice(weights_path=model_path / "model_price.pkl")
_toxicity_model = load(model_path / "model_toxicity.joblib")
_value_model = load_value_model(model_path / "model_value.pt", device="cpu")
_fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def price_heuristic(smiles: str) -> float:
    """
    Calculates expected market price of a molecule based on SMILES string.
    Uses a pre-instantiated model for efficiency.
    """
    price = _price_model.predict(smiles)  # type: ignore
    if type(price) is np.float32:
        price = float(min(1, price / 15))
        return price
    return 1.0


def toxicity_heuristic(smiles: str) -> float:
    """
    Calculate toxicity score from EToxPred model.
    """
    mol = Chem.MolFromSmiles(smiles)
    fp = _fp_generator.GetFingerprintAsNumPy(mol)
    prob = _toxicity_model.predict_proba(fp.reshape(1, -1))[0]
    score = float(prob[1])
    return score


def sustainability_heuristic(smiles: str) -> float:
    """
    Calculation of SAScore from RDKit implementation.
    """
    score = sascorer_optimized.calculateScore(Chem.MolFromSmiles(smiles))
    if score:
        score = float(score / 10)  # scale between 0 and 1
    else:
        score = 1.0
    return score


def value_heuristic(smiles: str) -> float:
    """
    Value function from Retro* paper.
    """
    mol = Chem.MolFromSmiles(smiles)
    fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = fp_generator.GetFingerprintAsNumPy(mol)
    fp = torch.tensor(fp).float().unsqueeze(0)
    value = _value_model(fp).item()
    # Scale value between 0 and 1 (assuming value is positive)
    value = float(min(1, value / 10))
    return value


COST_MAPPING = {
    "sustainability_cost": sustainability_heuristic,
    "scaleup_cost": price_heuristic,
    "toxicity_cost": toxicity_heuristic,
    "convergence_cost": value_heuristic,
    "retro_star_cost": value_heuristic,
}

if __name__ == "__main__":
    test_smiles = "CCO"
    import time

    start = time.time()
    print(f"Predicted price for {test_smiles}: {price_heuristic(test_smiles)}")
    end = time.time()
    print(f"Price prediction took {end - start:.4f} seconds")
    start = time.time()
    print(f"Predicted toxicity for {test_smiles}: {toxicity_heuristic(test_smiles)}")
    end = time.time()
    print(f"Toxicity prediction took {end - start:.4f} seconds")
    start = time.time()
    print(
        f"Predicted sustainability for {test_smiles}: {sustainability_heuristic(test_smiles)}"
    )
    end = time.time()
    print(f"Sustainability prediction took {end - start:.4f} seconds")
