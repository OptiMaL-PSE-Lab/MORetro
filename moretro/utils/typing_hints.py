from moretro.search.node_type import MolNode, RxnNode
from typing import TypeVar, Callable,  Any

# type var
NodeType = TypeVar("NodeType", bound=MolNode | RxnNode)

# base types
type Vector = list[float]
type Weights = list[list[float]]
type CostVector = tuple[float, ...]
type Nodes = MolNode | RxnNode
type WeightIndices = tuple[int, ...]
type Predictions = list[list[dict[str, Any]]]
type CostFunctions = list[Callable[[dict[str, Any]], float]]

# composite types storing paths
type Path = list[Nodes]
type PathCost = dict[CostVector, Path]
type SolutionPath = tuple[Path, WeightIndices]
type SolutionCost = dict[CostVector, SolutionPath]

# solution cost and weights
type ParetoCost = dict[CostVector, Weights]
type NewSolution = dict[CostVector, WeightIndices]

# inherent types
type NodesAndWeights[NodeType] = set[tuple[NodeType, WeightIndices]]
type MolNodeAndWeights = NodesAndWeights[MolNode]
type MolsAndWeights = NodesAndWeights[Nodes]
