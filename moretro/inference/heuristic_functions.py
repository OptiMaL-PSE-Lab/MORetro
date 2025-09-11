import warnings
import numpy as np
from pathlib import Path

from moretro.external.molprice import MolPrice

from sklearn.exceptions import InconsistentVersionWarning # type: ignore
warnings.filterwarnings(action='ignore', category=InconsistentVersionWarning)

#* Scale objectives between 0 and 10

# Instantiate the model once at module level to avoid overhead
model_path = Path(__file__).parent.parent / "models" / "MP_Morgan_hybrid.pkl"
_price_model = MolPrice(weights_path=model_path)

def price_heuristic(smiles: str) -> float: 
    """
    Calculates expected market price of a molecule based on SMILES string.
    Uses a pre-instantiated model for efficiency.
    """
    price = _price_model.predict(smiles) # type: ignore
    if type(price) is np.float32:
        price = float(max(0, price-5))
        return price
    return 10.0

def smiles_heuristic(smiles: str) -> float:
    """
    Heuristic based on SMILES length.
    Longer SMILES = higher cost.
    """
    return min(len(smiles) / 200, 1.0) * 10  # Scale to 0-10


COST_MAPPING = {
    "economic_cost": price_heuristic,
    "env_cost": smiles_heuristic,
}

if __name__ == "__main__":
    test_smiles = "CCO"
    print(f"Predicted price for {test_smiles}: {price_heuristic(test_smiles)}")
    print(f"SMILES heuristic for {test_smiles}: {smiles_heuristic(test_smiles)}")