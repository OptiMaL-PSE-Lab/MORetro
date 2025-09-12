# this file introduces the cost functions to calculate costs of reactions
# * All costs are scaled between 0 and 10

from typing import Any

import numpy as np
from rdkit import Chem

from moretro.utils.typing_hints import CostFunctions


def calculate_costs(
    prediction: dict[str, Any], cost_functions: CostFunctions
) -> list[float]:
    """
    Calculate multi-dimensional costs for a reaction.

    Parameters:
    -----------
    prediction : dict[str, Any]
        Prediction dictionary containing at minimum:
        - "rxn_smiles": str - Full reaction SMILES string
        - "reactants": list[str] - List of reactant SMILES strings
        - "template": str or list[str] - Reaction template in SMARTS format
        - "score": float - Model prediction score
    cost_functions : list[Callable]
        List of cost functions that take prediction dict and return float

    Returns:
    --------
    list[float]
        List of cost values, one per cost function
    """
    # TODO add temperature and reagents to predictions as well
    costs = []
    for cost_fn in cost_functions:
        cost = cost_fn(prediction)
        costs.append(cost)
    return costs


def atom_economy_cost(prediction: dict[str, Any]) -> float:
    """
    Atom economy cost based on the number of atoms in reactants vs products.
    Lower atom economy = higher cost.
    """
    rxn_smiles = prediction["rxn_smiles"]
    reactants_smiles, products_smiles = rxn_smiles.split(">>")

    # Calculate total heavy atoms in reactants
    reactant_atoms = 0
    for reactant in reactants_smiles.split("."):
        mol = Chem.MolFromSmiles(reactant.strip())
        if mol:
            reactant_atoms += mol.GetNumHeavyAtoms()

    # Calculate total heavy atoms in products
    product_atoms = 0
    for product in products_smiles.split("."):
        mol = Chem.MolFromSmiles(product.strip())
        if mol:
            product_atoms += mol.GetNumHeavyAtoms()

    atom_economy = product_atoms / reactant_atoms
    return (1 - atom_economy) * 10


def log_score(prediction: dict[str, Any]) -> float:
    """
    Logarithmic cost based on the model prediction score.
    Higher score = lower cost.
    """
    score = prediction["score"]
    return -np.log(score)  # Natural log


# Cost function mapping for easy configuration
COST_MAPPING = {
    "env_cost": atom_economy_cost,
    "economic_cost": log_score,
}
