import logging
import time
import gin
import numpy as np
import torch  # type: ignore
from typing import Callable
from collections import defaultdict


from moretro.search.mo_graph import MOGraph
from moretro.search.node_type import RxnNode, MolNode
from moretro.utils.typing_hints import MolNodeAndWeights, Nodes

# NOTE: For now, no dominance checks are implemented due to interdependent nature of problem

logger = logging.getLogger(__name__)


@gin.configurable(denylist=["target", "retro_model", "building_blocks", "heuristic_fns"])
class MOSearch:
    """
    Class for guiding the search process
    """

    def __init__(
        self,
        target: str,
        retro_model: Callable[[list[MolNode], int], dict],
        building_blocks: set[str],
        heuristic_fns: list[Callable[[str], float]],
        top_n: int,
        max_depth: int,
        iteration_budget: int,
        weight_iter_budget: int,
        time_budget: float = 0.0,
        weight_strategy: str = "it",  # "it" for iterative, "obj" for objective
    ):
        self.max_depth = max_depth
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

    def retro_expansion(self, nodes_and_weights: MolNodeAndWeights) -> None:
        """
        Expands node of graph by adding predictions of single-step model
        New nodes are added to the graph and values are updated accordingly

        Parameters:
            node (MolNode): The node to expand
        """
        nodes: list[MolNode] = []
        for node, _ in nodes_and_weights:
            if not self.can_expand_retro(node):
                logger.error(
                    f"Node {node} cannot be expanded in retrosynthetic direction. \n Aborting search..."
                )
                return
            else:
                nodes.append(node)

        predictions: dict[MolNode, list[dict]] = self.retro_model(nodes, self.top_n)
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

    def spawn_new_weights(self, num_iter: int) -> None:
        """
        Samples new weights to screen the Pareto front

        Parameters:
            num_iter (int): The number of iterations to sample new weights
        """
        if self.weight_strategy == "it":
            if num_iter % self.weight_iter_budget == 0:
                logger.info("Sampling new weights...")
                self.search_graph.reinitialize_graph()
        elif self.weight_strategy == "obj":
            logger.info("Sampling new weights as objectives are below threshold...")
            pass  #! Yet to implement objective-based search
        else:
            logger.error(
                f"Invalid weight strategy: {self.weight_strategy}. Use 'it' or 'obj'."
            )

    def choose_next_nodes(self) -> MolNodeAndWeights:
        """
        Select for each weight, which node to expand next.

        Returns:
            MolNodeAndWeights: A set of tuples containing nodes and their corresponding weight indices
        """

        open_nodes = []
        open_nodes_values = []
        for node in self.search_graph.open_nodes:
            open_nodes.append(node)
            open_nodes_values.append(node.total_value)

        open_values = np.array(open_nodes_values)  # dims are n_nodes x n_weights
        max_values = np.max(open_values, axis=0)
        is_max = open_values == max_values[None, :]
        indices_per_dim = []
        for i in range(open_values.shape[1]):
            dim_indices = np.where(is_max[:, i])[0].tolist()
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
        iter_counter = 0
        elapsed_time = 0
        start_time = time.time()
        while iter_counter < self.iteration_budget and elapsed_time < self.time_budget:
            torch.cuda.empty_cache()
            self.spawn_new_weights(iter_counter)
            if not self.search_graph.open_nodes:
                logger.info("No more open nodes to expand. Search completed.")
                break

            nodes_and_weights_to_expand = self.choose_next_nodes()
            self.retro_expansion(nodes_and_weights_to_expand)
            iter_counter += 1
            elapsed_time = time.time() - start_time