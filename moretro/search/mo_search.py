import logging
import time
import gin
import numpy as np
import torch  # type: ignore
from typing import Callable
from collections import defaultdict

from moretro.inference.retro_prediction import OneStepModel
from moretro.search.mo_graph import MOGraph
from moretro.search.node_type import RxnNode, MolNode
from moretro.utils.typing_hints import MolNodeAndWeights, Nodes

# NOTE: For now, no dominance checks are implemented due to interdependent nature of problem

logger = logging.getLogger(__name__)


@gin.configurable(
    denylist=["target", "retro_model", "building_blocks", "heuristic_fns"]
)
class MOSearch:
    """
    Class for guiding the search process
    """

    def __init__(
        self,
        target: str,
        retro_model: OneStepModel,
        building_blocks: set[str],
        heuristic_fns: list[Callable[[str], float]],
        top_n: int,
        max_depth: int,
        iteration_budget: int,
        weight_iter_budget: int,
        time_budget: float = 0.0,
        weight_strategy: str = "it",  # "it" for iterative, "obj" for objective
    ):
        self.max_depth = 2*max_depth
        self.retro_model = retro_model
        self.search_graph = MOGraph(
            target=target,
            building_blocks=building_blocks,
            heuristic_fns=heuristic_fns,
            weight_samples=gin.REQUIRED,  # type: ignore
            no_weights=gin.REQUIRED,  # type: ignore
            weight_initial=gin.REQUIRED,  # type: ignore
            include_extreme=gin.REQUIRED,  # type: ignore
        )
        self.top_n = top_n
        self.weight_strategy = weight_strategy
        self.weight_iter_budget = weight_iter_budget
        self.iteration_budget = iteration_budget
        self.time_budget = time_budget
        self.weights_open: list[bool] = [True] * self.search_graph.no_weights

    def can_expand_retro(self, node: Nodes) -> bool:
        """
        Check if the node can be expanded in the retrosynthetic direction
        Checks:
            1. Node is a MolNode
            2. Node has not reached max depth
            3. Node has not been expanded yet (is open)

        Parameters:
            node (MolNode): The node to check

        Returns:
            bool: True if the node can be expanded, False otherwise
        """
        return (
            isinstance(node, MolNode) and node.depth < self.max_depth and node.is_open
        )

    def retro_expansion(self, nodes_and_weights: MolNodeAndWeights) -> bool:
        """
        Expands node of graph by adding predictions of single-step model
        New nodes are added to the graph and values are updated accordingly

        Parameters:
            node (MolNode): The node to expand

        Returns:
            bool: True if early resampling is triggered (all weights blocked), False otherwise
        """
        nodes: list[MolNode] = []
        nodes_and_weights_copy = nodes_and_weights.copy()
        for node, weight in nodes_and_weights_copy:
            if not self.can_expand_retro(node) and node.depth >= self.max_depth:
                for w in weight:
                    if self.weights_open[w]:
                        self.weights_open[w] = False
                        logger.info(
                            f"Node {node.smiles} cannot be expanded with depth {int(node.depth / 2)} and a pre-defined max depth {int(self.max_depth / 2)}."
                        )
                        logger.warning(f"Weight {w} is now blocked from expansion until resampling.")
                nodes_and_weights.remove((node, weight))
            elif not self.can_expand_retro(node):
                logger.critical("Critical error in expansion logic. This is a bug.")
            else:
                nodes.append(node)
        
        if not nodes:
            logger.warning("All weights selected for expansion cannot expand further. Early resampling triggered.")
            return True

        smiles = [node.smiles for node in nodes]
        predictions = self.retro_model.predict(
            smiles, self.top_n
        )  # * adds predictions with costs
        predictions = {node: preds for node, preds in zip(nodes, predictions)}
        new_nodes_and_weights = self.search_graph.expand_graph(
            predictions, nodes_and_weights
        )
        if not new_nodes_and_weights:
            logger.warning(
                "No new nodes were generated during expansion. Retrosynthesis tree was not expanded further."
            )
        else:
            for new_node, _ in new_nodes_and_weights:
                if self.can_expand_retro(new_node):
                    self.search_graph.open_nodes.add(new_node)  # type: ignore

            _ = self.search_graph.update_values(
                new_nodes_and_weights.union(nodes_and_weights)
            )

        return False

    def spawn_new_weights(self, num_iter: int, early_resampling: bool) -> int:
        """
        Samples new weights to screen the Pareto front

        Parameters:
            num_iter (int): The number of iterations to sample new weights
            early_resampling (bool): Whether the resampling is triggered early due to all weights being blocked
        
        Returns:
            int: 1 to reinit weight counter, 0 otherwise
        """
        if self.weight_strategy == "it":
            if num_iter == self.weight_iter_budget + 1 or early_resampling:
                logger.info("Sampling new weights...")
                if self.search_graph.weights_open.shape[0] < self.search_graph.no_weights:
                    return -100 # exit search 
                self.search_graph.reinitialize_graph()
                return 1
        elif self.weight_strategy == "obj":
            logger.info("Sampling new weights as objectives are below threshold...")
            # TODO Yet to implement objective-based search
            return 0
        else:
            logger.error(
                f"Invalid weight strategy: {self.weight_strategy}. Use 'it' or 'obj'."
            )
            return 0
        return num_iter

    def choose_next_nodes(self) -> MolNodeAndWeights:
        """
        Select for each weight, which node to expand next.

        Returns:
            MolNodeAndWeights: A set of tuples containing nodes and their corresponding weight indices
        """

        # Convert set to sorted list for consistent ordering
        open_nodes = sorted(list(self.search_graph.open_nodes), key=lambda x: x.smiles)
        open_nodes_values = []
        for node in open_nodes:
            open_nodes_values.append(node.total_value)

        open_values = np.array(open_nodes_values)  # dims are n_nodes x n_weights
        min_values = np.min(open_values, axis=0)
        is_min = open_values == min_values[None, :]
        indices_per_dim = []
        for i in range(open_values.shape[1]):
            dim_indices = np.where(is_min[:, i])[0].tolist()
            indices_per_dim.append(dim_indices)

        # Find which weight dimensions want to expand the same nodes
        identical_groups = defaultdict(list)

        for i, indices in enumerate(indices_per_dim):
            # Use sorted tuple as key to group identical lists
            key = tuple(sorted(indices))
            identical_groups[key].append(i)

        nodes_and_weights_to_expand = set()
        for key, dims in identical_groups.items():
            # key contains the node indices that these weight dimensions prefer
            # dims contains the weight dimension indices that prefer these nodes

            node_indices = list(key)
            # Assign different nodes to weights when possible, cycling through available nodes
            for i in range(len(dims)):
                node_idx = node_indices[i % len(node_indices)]
                nodes_and_weights_to_expand.add((open_nodes[node_idx], tuple(dims)))

        return nodes_and_weights_to_expand

    def run_mo_search(self) -> None:
        """
        Runs the multi-objective search process iteratively.
            1. Select n nodes to expand based on n weights
            2. Expand the selected nodes
            3. Check if new weights should be sampled
            4. Repeat until iteration budget is reached or Pareto front is stable
        """
        logger.info("Starting multi-objective search process...")
        iter_counter = 1
        weight_iter = 1
        elapsed_time = 0
        start_time = time.time()
        while iter_counter < self.iteration_budget and elapsed_time < self.time_budget:
            torch.cuda.empty_cache()
            weight_iter = self.spawn_new_weights(weight_iter, early_resampling=False)
            break_condition = not self.search_graph.open_nodes or weight_iter < 0
            if break_condition:
                if not self.search_graph.open_nodes:
                    logger.info("No open nodes left to expand.")
                else:
                    logger.info("All weights have been sampled, no more weights to explore.")
                logger.info("Search process completed.")
                break

            nodes_and_weights_to_expand = self.choose_next_nodes()

            early_resampling = self.retro_expansion(nodes_and_weights_to_expand)
            if early_resampling:
                weight_iter = self.spawn_new_weights(iter_counter, early_resampling=True)
                self.weights_open = [True] * self.search_graph.no_weights
            iter_counter += 1
            weight_iter += 1
            elapsed_time = time.time() - start_time
