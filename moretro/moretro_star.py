import logging
import logging.config as conf
import pprint
import tempfile
from pathlib import Path as PathLib
from typing import Any

import gin
import graphviz
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw

from moretro.inference.retro_prediction import OneStepModel
from moretro.search.mo_search import MOSearch
from moretro.search.node_type import MolNode, RxnNode
from moretro.utils.prepare_models import (
    prepare_heuristic_fns,
    prepare_starting_mols,
)
from moretro.utils.typing_hints import Path

# set up logging in this main file
conf.fileConfig("moretro/configs/logging.conf")
logger = logging.getLogger("moretro")


class MORetro:
    def __init__(self, target: str):
        retro_model = OneStepModel(gin.REQUIRED)  # type: ignore
        building_blocks = prepare_starting_mols(gin.REQUIRED)  # type: ignore
        heuristic_fns = prepare_heuristic_fns(gin.REQUIRED)  # type: ignore
        self.mo_search = MOSearch(
            target, retro_model, building_blocks, heuristic_fns, gin.REQUIRED
        )  # type: ignore
        self.target = target

    def search(self):
        """
        Run the multi-objective retrosynthesis search and plot results
        """
        try:
            self.mo_search.run_mo_search()
            logger.info("Search completed.")
        except KeyboardInterrupt:
            logger.warning("Search interrupted by user.")
        finally:
            # TODO improve this
            logger.info("Creating plots for target and saving in ./figs directory")
            self.visualize_all_solutions()
            self.plot_pareto_front()
            solution_summary = self.get_solution_summary()
            logger.info(
                "Solution summary: \n" + pprint.pformat(solution_summary, indent=2)
            )

    def _visualize_path(self, path: Path, output_path: str, title: str):
        """
        Directly visualize a solution path using the graph structure.
        Simple and efficient approach that works directly with your path-based storage.

        Parameters
        ----------
        path : Path
            List of nodes representing the synthesis path
        output_path : str
            File path to save the visualization
        title : str
            Title for the visualization
        """
        dot = graphviz.Digraph(format="png")
        dot.attr(label=title, labelloc="t", fontsize="16")

        with tempfile.TemporaryDirectory() as temp_img_dir:
            # Draw all nodes in the path
            for node in path:
                if isinstance(node, MolNode):
                    self._add_mol_node(node, dot, temp_img_dir)
                elif isinstance(node, RxnNode):
                    self._add_rxn_node(node, dot)

            # Add edges based on the graph structure
            graph = self.mo_search.search_graph.graph
            for node in path:
                for successor in graph.successors(node):
                    if (
                        successor in path
                    ):  # Only connect nodes that are in this solution path
                        dot.edge(node.smiles, successor.smiles, color="darkgrey")

            dot.render(output_path, cleanup=True)

    def _add_mol_node(self, node: MolNode, dot: graphviz.Digraph, temp_img_dir: str):
        """Add a molecule node to the graph."""
        mol = Chem.MolFromSmiles(node.smiles)
        escaped = node.smiles.replace("/", "_").replace("\\", "_")
        file_path = f"{temp_img_dir}/{escaped}.png"
        Draw.MolToFile(mol, file_path, size=(200, 200))

        # Determine color based on node properties
        if getattr(node, "is_target", False):
            color = "lightsalmon"
        elif node.is_known:  # Building block
            color = "springgreen3"
        else:  # Intermediate
            color = "royalblue"

        dot.node(
            node.smiles,
            label="",
            image=file_path,
            shape="box",
            color=color,
            penwidth="2",
        )

    def _add_rxn_node(self, node: RxnNode, dot: graphviz.Digraph):
        """Add a reaction node to the graph."""
        dot.node(
            node.smiles,
            label="",
            shape="box",
            style="rounded",
            color="lightgoldenrod1",
            penwidth="2",
        )

    def visualize_pareto_solutions(self, output_dir: str):
        """
        Visualize all Pareto-optimal synthesis routes using direct path visualization.

        Parameters
        ----------
        output_dir : str
            Directory to save the visualization files
        """
        if not PathLib(output_dir).exists():
            PathLib(output_dir).mkdir(parents=True, exist_ok=True)

        pareto_front = self.mo_search.search_graph.pareto_front
        solution_cost = self.mo_search.search_graph.solution_cost

        logger.info(f"Found {len(pareto_front)} Pareto-optimal solutions")

        for i, (cost_vector, _) in enumerate(pareto_front.items()):
            if cost_vector in solution_cost:
                path, _ = solution_cost[cost_vector]
                title = f"Pareto Solution {i + 1}\\nCost: {[f'{c:.3f}' for c in cost_vector]}"
                output_path = str(PathLib(output_dir) / f"pareto_route_{i + 1}")
                self._visualize_path(path, output_path, title)
                logger.debug(f"Saved Pareto route {i + 1} to {output_path}.png")

    def visualize_dominated_solutions(self, output_dir: str, max_solutions: int = 100):
        """
        Visualize dominated (non-Pareto) synthesis routes using direct path visualization.

        Parameters
        ----------
        output_dir : str
            Directory to save the visualization files
        max_solutions : int
            Maximum number of dominated solutions to visualize
        """
        if not PathLib(output_dir).exists():
            PathLib(output_dir).mkdir(parents=True, exist_ok=True)

        pareto_front = self.mo_search.search_graph.pareto_front
        solution_cost = self.mo_search.search_graph.solution_cost

        # Find dominated solutions
        dominated_solutions = [
            (cost, path, idx)
            for cost, (path, idx) in solution_cost.items()
            if cost not in pareto_front
        ]

        logger.info(f"Found {len(dominated_solutions)} dominated solutions")
        dominated_solutions = dominated_solutions[:max_solutions]
        logger.debug(
            f"Saving a max of {max_solutions} dominated solutions at {output_dir}"
        )

        for i, (cost_vector, path, _) in enumerate(dominated_solutions):
            title = f"Dominated Solution {i + 1}\\nCost: {[f'{c:.3f}' for c in cost_vector]}"
            output_path = str(PathLib(output_dir) / f"dominated_route_{i + 1}")
            self._visualize_path(path, output_path, title)

    def visualize_all_solutions(self, output_dir: str = "figs"):
        """
        Visualize both Pareto-optimal and dominated synthesis routes using direct visualization.

        Parameters
        ----------
        output_dir : str
            Base directory to save the visualization files
        """
        # Create target-specific folder structure
        safe_target_name = self._safe_smiles_dirname(self.target)
        target_dir = str(PathLib(output_dir) / safe_target_name)

        pareto_dir = str(PathLib(target_dir) / "pareto")
        dominated_dir = str(PathLib(target_dir) / "dominated")

        logger.info("Visualizing Pareto-optimal solutions...")
        self.visualize_pareto_solutions(pareto_dir)

        logger.info("Visualizing dominated solutions...")
        self.visualize_dominated_solutions(dominated_dir)

        logger.info(f"All visualizations saved to {target_dir}")

    def get_solution_summary(self) -> dict[str, Any]:
        """
        Get a summary of all discovered solutions.

        Returns
        -------
        dict[str, Any]
            Summary containing counts and cost ranges
        """
        pareto_front = self.mo_search.search_graph.pareto_front
        solution_cost = self.mo_search.search_graph.solution_cost

        pareto_costs = list(pareto_front.keys())
        all_costs = list(solution_cost.keys())
        dominated_costs = [cost for cost in all_costs if cost not in pareto_front]

        summary: dict[str, Any] = {
            "total_solutions": len(all_costs),
            "pareto_solutions": len(pareto_costs),
            "dominated_solutions": len(dominated_costs),
        }

        if all_costs:
            all_costs_array = np.array(all_costs)
            summary["cost_ranges"] = {
                f"objective_{i}": {
                    "min": round(float(all_costs_array[:, i].min()), 2),
                    "max": round(float(all_costs_array[:, i].max()), 2),
                    "mean": round(float(all_costs_array[:, i].mean()), 2),
                }
                for i in range(all_costs_array.shape[1])
            }

        return summary

    def plot_pareto_front(
        self,
        output_path: str | None = None,
        figsize: tuple[int, int] = (10, 6),
        show_weights: bool = True,
        weight_fontsize: int = 10,
    ):
        """
        Plot the Pareto front in 2D or 3D with weight labels for each point.

        Parameters
        ----------
        output_path : str
            File path to save the plot (without extension). If None, will save to
            figs/{target_smiles}/pareto_front
        figsize : tuple[int, int]
            Figure size (width, height)
        show_weights : bool
            Whether to show weight vectors as labels
        weight_fontsize : int
            Font size for weight labels
        """
        pareto_front = self.mo_search.search_graph.pareto_front

        if not pareto_front:
            logger.warning("No Pareto solutions found to plot.")
            return

        # Set default output path if not provided
        if output_path is None:
            safe_target_name = self._safe_smiles_dirname(self.target)
            output_path = f"figs/{safe_target_name}/pareto_front"

        if not PathLib(output_path).parent.exists():
            PathLib(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Extract cost vectors and weights
        cost_vectors = list(pareto_front.keys())
        weights = list(pareto_front.values())

        costs_array = np.array(cost_vectors)
        n_objectives = costs_array.shape[1]

        if n_objectives < 2:
            logger.warning("Need at least 2 objectives to plot Pareto front.")
            return
        elif n_objectives == 2:
            self._plot_pareto_2d(
                costs_array,
                weights,
                output_path,
                figsize,
                show_weights,
                weight_fontsize,
            )
        elif n_objectives == 3:
            self._plot_pareto_3d(
                costs_array,
                weights,
                output_path,
                figsize,
                show_weights,
                weight_fontsize,
            )
        else:
            logger.warning(
                f"Plotting not supported for {n_objectives} objectives. Plotting first 3 dimensions."
            )

    def _plot_pareto_2d(
        self,
        costs_array: np.ndarray,
        weights: list,
        output_path: str,
        figsize: tuple[int, int],
        show_weights: bool,
        weight_fontsize: int,
    ):
        """Plot 2D Pareto front."""
        plt.figure(figsize=figsize)

        # Plot Pareto points
        plt.scatter(
            costs_array[:, 0],
            costs_array[:, 1],
            c="red",
            s=100,
            alpha=0.7,
            edgecolors="black",
            linewidth=1,
            label="Pareto Solutions",
        )

        # Add weight labels if requested
        if show_weights:
            for cost, weight_list in zip(costs_array, weights, strict=True):
                # Format weight vector(s) for display
                weight_parts = []
                weight_str = ""
                for weight in weight_list:
                    formatted_weights = [f"{w:.2f}" for w in weight]
                    weight_parts.append(f"[{', '.join(formatted_weights)}]")

                weight_str += "; ".join(weight_parts)
                plt.annotate(
                    weight_str,
                    (cost[0], cost[1]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=weight_fontsize,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                )

        plt.xlabel("Objective 1", fontsize=12)
        plt.ylabel("Objective 2", fontsize=12)
        plt.title("Pareto Front (2D)", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Connect points to show Pareto front
        sorted_indices = np.argsort(costs_array[:, 0])
        sorted_costs = costs_array[sorted_indices]
        plt.plot(
            sorted_costs[:, 0],
            sorted_costs[:, 1],
            "r--",
            alpha=0.5,
            linewidth=1,
        )

        plt.tight_layout()
        plt.savefig(f"{output_path}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{output_path}.pdf", bbox_inches="tight")
        plt.close()

    def _plot_pareto_3d(
        self,
        costs_array: np.ndarray,
        weights: list,
        output_path: str,
        figsize: tuple[int, int],
        show_weights: bool,
        weight_fontsize: int,
    ):
        """Plot 3D Pareto front."""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Plot Pareto points
        ax.scatter(
            costs_array[:, 0],
            costs_array[:, 1],
            costs_array[:, 2],
            c="red",
            s=100,  # type: ignore
            alpha=0.7,
            edgecolors="black",
            label="Pareto Solutions",
        )

        # Add weight labels if requested
        if show_weights:
            for cost, weight_list in zip(costs_array, weights, strict=True):
                # Format weight vector(s) for display
                weight_parts = []
                weight_str = ""
                for weight in weight_list:
                    formatted_weights = [f"{w:.2f}" for w in weight]
                    weight_parts.append(f"[{', '.join(formatted_weights)}]")

                weight_str += "; ".join(weight_parts)
                # Add text annotation
                ax.text(cost[0], cost[1], cost[2], weight_str, fontsize=weight_fontsize)  # type: ignore

        ax.set_xlabel("Objective 1", fontsize=12)
        ax.set_ylabel("Objective 2", fontsize=12)
        ax.set_zlabel("Objective 3", fontsize=12)  # type: ignore
        ax.set_title("Pareto Front (3D)", fontsize=14, fontweight="bold")
        ax.legend()

        # Add grid
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_path}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{output_path}.pdf", bbox_inches="tight")
        plt.close()

    def _safe_smiles_dirname(self, smiles: str) -> str:
        """
        Convert SMILES string to a safe directory name by replacing problematic characters.

        Parameters
        ----------
        smiles : str
            SMILES string to convert

        Returns
        -------
        str
            Safe directory name
        """
        # Replace problematic characters with safe alternatives
        safe_name = smiles.replace("/", "_slash_")
        safe_name = safe_name.replace("\\", "_backslash_")
        safe_name = safe_name.replace(":", "_colon_")
        safe_name = safe_name.replace("*", "_star_")
        safe_name = safe_name.replace("?", "_question_")
        safe_name = safe_name.replace('"', "_quote_")
        safe_name = safe_name.replace("<", "_lt_")
        safe_name = safe_name.replace(">", "_gt_")
        safe_name = safe_name.replace("|", "_pipe_")
        return safe_name


if __name__ == "__main__":
    gin.parse_config_file("moretro/configs/search_config.gin")
    target_smiles = "CC(C)c1ccc(-n2nc(O)c3c(=O)c4ccc(Cl)cc4[nH]c3c2=O)cc1"  # Example target: Aspirin
    moretro = MORetro(target_smiles)
    moretro.search()
