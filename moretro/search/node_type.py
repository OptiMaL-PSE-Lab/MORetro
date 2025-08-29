from __future__ import annotations

import logging
import numpy as np
from itertools import product
from typing import Callable
from dataclasses import dataclass, field
from rdkit import Chem

type Vector = list[float]
type Path = list[MolNode | RxnNode]
type PathCost = dict[tuple[float, ...], Path]


logger = logging.getLogger(__name__)


def zero_vector(length: int) -> list[float]:
    return [0.0] * length


@dataclass(frozen=False)
class MolNode:
    """
    Class to represent a molecule node in the multi-objective retrosynthesis search graph.
    This node is an OR node, meaning it only needs one reaction (child node) to be successful

    Parameters:
    -----------
    smiles : str
        Molecule in canonical SMILES format
    heuristic_fns : list[Callable[[str], float]]
        List of heuristic functions that calculate objective values for this molecule.
    depth : int
        Depth of the node in the search tree. Root (target) molecule has depth 0,
    is_known : bool
        Whether this molecule is available in the building blocks (known starting materials).
    rxn_no : Vector (list[float])
        Reaction number vector containing scalar values for each weight group.
    total_value : Vector (list[float])
        Total value vector propagated from parent nodes, one value for each weight group.
    success_cost : PathCost
        Dictionary mapping cost vectors (as tuples) to the corresponding synthesis paths.
        Key: tuple of objective costs, Value: list of nodes in the path.
    success : bool
        Whether this node has at least one successful synthesis path to building blocks.
    success_cost: PathCost
        Dictionary mapping cost vectors for successful synthesis to corresponding path + weights
        Key: tuple of objective costs, Value: tuple of nodes in the path with weight indices
    is_open : bool
        Whether this node is available for expansion in the search.
    zero_bound : bool
        Whether to use zero lower bounds for known molecules. If True, known molecules
    value_estimates : Vector (list[float])
        Heuristic-based estimates for each objective (computed in __post_init__).

    success_cost_estimate : Vector (list[float])
    """

    smiles: str = field(compare=True)
    heuristic_fns: list[Callable[[str], float]]
    depth: int
    is_known: bool
    rxn_no: Vector = field(default_factory=list)
    total_value: Vector = field(default_factory=list)
    success: bool = False
    success_cost: PathCost = field(default_factory=dict)
    is_open: bool = True
    zero_bound: bool = True
    is_target: bool = False  # whether this is the target molecule

    def __post_init__(self) -> None:
        self.h_length: int = len(self.heuristic_fns)
        self.value_estimates = self._calculate_heuristics()

        if self.is_known:  # initiate the objectives
            self.is_open = False
            if self.zero_bound:
                self.success_cost_estimate = zero_vector(len(self.value_estimates))
            else:
                self.success_cost_estimate = self.value_estimates
        self.smiles = self._canonicalize_smiles(self.smiles)

    def _canonicalize_smiles(self, smiles: str) -> str:
        return Chem.CanonSmiles(smiles)

    def _calculate_heuristics(self) -> list[float]:
        objectives = []
        for heuristic in self.heuristic_fns:
            objectives.append(heuristic(self.smiles))
        return objectives

    def objectives_to_scalar(self, weights: np.ndarray) -> np.ndarray:
        """
        Convert the objectives to a scalar using current weights and initialize rxn_no
        """
        rxn_no = []
        for weight in weights:
            rxn_no.append(np.dot(np.array(self.value_estimates), weight))
        return np.array(rxn_no)

    def uppropagate(self, children: list[RxnNode], weights: np.ndarray) -> bool:
        """
        Propagate costs upward from reaction children (OR logic).

        Parameters
        ----------
        children : list[RxnNode]
            Reaction node children (synthesis routes).
        weights : np.ndarray
            Weight matrix for scalarization.

        Returns
        -------
        bool
            True if node attributes were modified.
        """
        new_success_cost = dict()
        success = False
        if self.is_known:  # known building block
            no_weights, _ = weights.shape
            if self.zero_bound:
                new_rxn_no = np.zeros(no_weights)
            else:
                new_rxn_no = self.objectives_to_scalar(weights)
            success = True
            new_success_cost: PathCost = {tuple(self.success_cost_estimate): [self]}
        elif self.is_open:  # tip node of tree which is not a building block
            new_rxn_no = self.objectives_to_scalar(weights)
        else:
            if len(children) > 0:  # interior node with children
                children_rxn_no = np.array(
                    [child.rxn_no for child in children]
                )  # shape: (n_children, n_objectives)
                new_rxn_no = np.min(children_rxn_no, axis=0)
                success = any(child.success for child in children)
                if success:
                    new_success_cost = self.track_success_cost(children)

            else:  # no valid expansion
                new_rxn_no = np.full(weights.shape[0], np.inf)

        new_rxn_no = new_rxn_no.tolist()  # convert to list for comparison
        if (
            self.rxn_no != new_rxn_no or new_success_cost or self.success != success
        ):  # if any of the values changed, update the node and return bool True
            self.rxn_no = new_rxn_no
            self.success = success
            self.success_cost.update(new_success_cost)
            return True
        return False

    def downpropagate(self, parents: list[RxnNode]) -> bool:
        """
        Propagate total values downward from parent reactions.

        Parameters
        ----------
        parents : list[RxnNode]
            Parent reaction nodes producing this molecule.

        Returns
        -------
        bool
            True if total_value was updated.
        """
        if len(parents) == 0:  # product node with no parents
            new_total_value = self.rxn_no
        else:
            new_total_value = np.min(
                np.array([p.total_value for p in parents]), axis=0
            ).tolist()  # the total value is the minmum of the parents' total values
        if self.total_value != new_total_value:
            self.total_value = new_total_value
            return True
        return False

    def track_success_cost(self, children: list[RxnNode]) -> PathCost:
        """
        Track successful synthesis paths from child reactions.

        Parameters
        ----------
        children : list[RxnNode]
            Child reaction nodes with successful paths.
        Returns
        -------
        PathCost
            Updated success cost dictionary.
        """
        new_success_cost = dict()
        all_costs_successors = {
            cost: successor
            for child in children
            for cost, successor in child.success_cost.items()
            if successor  # Filter out empty successors
        }

        current_success_cost_keys = set(self.success_cost.keys())

        for cost, successor in all_costs_successors.items():
            # check if the cost is already in the success_cost
            if cost in current_success_cost_keys:
                continue

            new_path = successor + [self]
            new_success_cost[cost] = new_path
        return new_success_cost

    def __hash__(self) -> int:
        return id(self)


@dataclass(frozen=False)
class RxnNode:
    """
    Reaction node in multi-objective retrosynthesis search graph (AND node).

    Parameters
    ----------
    smiles : str
        Reaction in SMILES format.
    template : str
        Reaction template in SMARTS format.
    reagents : str
        Required reagents/catalysts.
    temp : float
        Reaction temperature in Kelvin.
    depth : int
        Node depth in search tree.
    cost : Vector
        Multi-dimensional reaction cost.
    weight_length : int
        Number of weight samples.
    """

    smiles: str
    template: str
    reagents: str
    temp: float
    depth: int
    cost: Vector  # Actual cost of reaction in n dimensions
    weight_length: int
    # running_cost: list[np.ndarray] = field(
    # default_factory=list
    # )  # cost of all previous reactions in path
    total_value: Vector = field(default_factory=list)  # one for each group of weight
    rxn_no: Vector = field(default_factory=list)  # one for each group of weights
    success_cost: PathCost = field(default_factory=dict)
    success: bool = False
    _delta_offset: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if not self.cost:
            logger.error("Reaction cost cannot be empty!")
            raise ValueError("Reaction cost must be provided")

        reaction_hash = hash((self.smiles, self.template, self.reagents))
        normalized_hash = (abs(reaction_hash) % 100) + 1
        self._delta_offset = normalized_hash * 1e-15
        self.true_cost = self.cost.copy()
        self.cost = [c + self._delta_offset for c in self.cost]  # ensure unique costs
        self.rxn_no = zero_vector(self.weight_length)
        self.total_value = zero_vector(self.weight_length)

    def uppropagate(self, children: list[MolNode], weights: np.ndarray) -> bool:
        """
        Propagate costs upward from molecule children (AND logic).

        Parameters
        ----------
        children : list[MolNode]
            Molecule node children (reactants).
        weights : np.ndarray
            Weight matrix for scalarization.

        Returns
        -------
        bool
            True if node attributes were modified.

        """
        new_success = False
        if not children:
            logger.error(
                "Reaction node must have at least one child to propagate from!"
            )
            raise ValueError("No children provided for RxnNode")
        new_success = all([child.success for child in children])
        new_rxn_no = np.zeros(len(self.rxn_no))
        new_success_cost = dict()

        # Sum up costs from all children
        for child in children:
            assert len(child.rxn_no) != 0, "Rxn_no for MolNode should not be empty"
            new_rxn_no += np.array(child.rxn_no)
        # add reaction cost with each weight combination
        rxn_cost = weights @ self.true_cost
        new_rxn_no += rxn_cost

        if new_success:
            new_success_cost = self.track_success_cost(children)

        new_rxn_no = new_rxn_no.tolist()
        if self.rxn_no != new_rxn_no or new_success_cost or self.success != new_success:
            self.rxn_no = new_rxn_no
            self.success = new_success
            self.success_cost.update(new_success_cost)
            return True
        return False

    def downpropagate(self, parent: MolNode) -> bool:
        """
        Propagate total values downward from parent molecule.

        Parameters
        ----------
        parent : MolNode
            Parent molecule node this reaction produces.

        Returns
        -------
        bool
            True if total_value was updated.
        """
        new_total_value = (
            np.array(self.rxn_no)
            - np.array(parent.rxn_no)
            + np.array(parent.total_value)
        )
        new_total_value = new_total_value.tolist()
        if self.total_value != new_total_value:
            self.total_value = new_total_value
            return True
        return False

    def track_success_cost(self, children: list[MolNode]) -> PathCost:
        """
        Track synthesis paths from child molecules (AND logic).

        Parameters
        ----------
        children : list[MolNode]
            Child molecule nodes (reactants).

        Returns
        -------
        PathCost
            Updated success cost dictionary with path combinations.
        """
        new_success_cost = dict()
        current_success_cost_keys = set(self.success_cost.keys())

        children_costs = [list(child.success_cost.keys()) for child in children]
        for cost_combination in product(*children_costs):
            successor_nodes = []

            # Collect all successor nodes from children
            for i, cost in enumerate(cost_combination):
                child_successor = children[i].success_cost[cost]

                if isinstance(child_successor, tuple):  # for typing consistency
                    child_nodes, _ = child_successor
                    successor_nodes.extend(child_nodes)
                else:
                    # Path format: list of nodes
                    successor_nodes.extend(child_successor)

            # Calculate total cost: sum of children costs + reaction cost
            children_total_cost = np.sum(np.array(cost_combination), axis=0)
            reaction_total_cost = children_total_cost + np.array(self.cost)
            reaction_total_cost = tuple(reaction_total_cost.tolist())

            # Check if this cost already exists
            if reaction_total_cost not in current_success_cost_keys:
                # Add this reaction to the path and store
                successor_nodes.append(self)
                new_success_cost[reaction_total_cost] = successor_nodes

        return new_success_cost

    def __hash__(self) -> int:
        return id(self)
