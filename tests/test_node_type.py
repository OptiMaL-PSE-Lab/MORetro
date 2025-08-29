"""
Unit tests for MolNode and RxnNode classes in node_type.py

Tests cover core functionalities including:
- Node initialization and properties
- Value propagation (up and down)
- Success cost tracking
- Heuristic calculations
- Path management
"""

import pytest
import numpy as np
from unittest.mock import Mock
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from moretro.search.node_type import MolNode, RxnNode, zero_vector


class TestMolNode:
    """Test cases for MolNode class"""

    @pytest.fixture
    def heuristic_fns(self):
        """Create mock heuristic functions"""

        def h1(smiles):
            return 1.0

        def h2(smiles):
            return 2.0

        return [h1, h2]

    @pytest.fixture
    def simple_mol_node(self, heuristic_fns):
        """Create a simple MolNode for testing"""
        return MolNode(
            smiles="CCO", heuristic_fns=heuristic_fns, depth=1, is_known=False
        )

    @pytest.fixture
    def known_mol_node(self, heuristic_fns):
        """Create a known (building block) MolNode"""
        return MolNode(smiles="CC", heuristic_fns=heuristic_fns, depth=2, is_known=True)

    def test_initialization_basic(self, heuristic_fns):
        """Test basic MolNode initialization"""
        node = MolNode(
            smiles="CCO", heuristic_fns=heuristic_fns, depth=1, is_known=False
        )

        assert node.smiles == "CCO"
        assert node.depth == 1
        assert not node.is_known
        assert node.is_open
        assert not node.success
        assert node.h_length == 2
        assert node.value_estimates == [1.0, 2.0]
        assert node.success_cost == {}

    def test_known_molecule_initialization(self, heuristic_fns):
        """Test initialization of known building block"""
        node = MolNode(
            smiles="CC",
            heuristic_fns=heuristic_fns,
            depth=0,
            is_known=True,
            zero_bound=True,
        )

        assert not node.is_open  # Known molecules are not open for expansion
        assert node.success_cost_estimate == [0.0, 0.0]  # Zero bound

    def test_known_molecule_no_zero_bound(self, heuristic_fns):
        """Test known molecule without zero bound"""
        node = MolNode(
            smiles="CC",
            heuristic_fns=heuristic_fns,
            depth=0,
            is_known=True,
            zero_bound=False,
        )

        assert node.success_cost_estimate == [1.0, 2.0]  # Heuristic values

    def test_objectives_to_scalar(self, simple_mol_node):
        """Test conversion of objectives to scalar values"""
        weights = np.array([[0.5, 0.5], [1.0, 0.0]])
        result = simple_mol_node.objectives_to_scalar(weights)
        expected = np.array([1.5, 1.0])  # [0.5*1.0 + 0.5*2.0, 1.0*1.0 + 0.0*2.0]
        np.testing.assert_array_equal(result, expected)

    def test_uppropagate_known_molecule(self, known_mol_node):
        """Test uppropagation for known building block"""
        weights = np.array([[0.6, 0.4], [0.3, 0.7]])

        result = known_mol_node.uppropagate([], weights)

        assert result  # Should return True for changes
        assert known_mol_node.success
        assert known_mol_node.rxn_no == [0.0, 0.0]  # Zero bound
        assert (
            tuple(known_mol_node.success_cost_estimate) in known_mol_node.success_cost
        )

    def test_uppropagate_open_node(self, simple_mol_node):
        """Test uppropagation for open (leaf) node"""
        weights = np.array([[0.5, 0.5]])

        result = simple_mol_node.uppropagate([], weights)

        assert result
        assert (
            not simple_mol_node.success
        )  # Open node without children is not successful
        assert simple_mol_node.rxn_no == [1.5]  # 0.5*1.0 + 0.5*2.0

    def test_uppropagate_with_children(self, simple_mol_node):
        """Test uppropagation with successful children"""
        # Create mock children
        child1 = Mock(spec=RxnNode)
        child1.rxn_no = [2.0, 3.0]
        child1.success = True
        child1.success_cost = {(1.0, 1.5): ["path1"]}

        child2 = Mock(spec=RxnNode)
        child2.rxn_no = [1.5, 4.0]
        child2.success = False
        child2.success_cost = {}

        simple_mol_node.is_open = False
        weights = np.array([[1.0, 0.0], [0.0, 1.0]])

        result = simple_mol_node.uppropagate([child1, child2], weights)

        assert result
        assert simple_mol_node.success  # At least one child is successful
        assert simple_mol_node.rxn_no == [1.5, 3.0]  # Min of children

    def test_downpropagate_no_parents(self, simple_mol_node):
        """Test downpropagation for node with no parents"""
        simple_mol_node.rxn_no = [2.0, 3.0]

        result = simple_mol_node.downpropagate([])

        assert result
        assert simple_mol_node.total_value == [2.0, 3.0]

    def test_downpropagate_with_parents(self, simple_mol_node):
        """Test downpropagation with parent reactions"""
        parent1 = Mock(spec=RxnNode)
        parent1.total_value = [1.0, 2.0]

        parent2 = Mock(spec=RxnNode)
        parent2.total_value = [0.5, 3.0]

        result = simple_mol_node.downpropagate([parent1, parent2])

        assert result
        assert simple_mol_node.total_value == [0.5, 2.0]  # Min of parents

    def test_track_success_cost_basic(self, simple_mol_node):
        """Test basic success cost tracking"""
        child = Mock(spec=RxnNode)
        child.success_cost = {(2.0, 3.0): ["node1", "node2"]}

        simple_mol_node.success_cost = {}

        result = simple_mol_node.track_success_cost([child])

        assert (2.0, 3.0) in result
        assert result[(2.0, 3.0)] == ["node1", "node2", simple_mol_node]

    def test_track_success_cost_target_node(self, simple_mol_node):
        """Test success cost tracking for target node"""
        simple_mol_node.is_target = True
        child = Mock(spec=RxnNode)
        child.success_cost = {(1.0, 2.0): ["node1"]}

        result = simple_mol_node.track_success_cost([child])

        assert (1.0, 2.0) in result
        path = result[(1.0, 2.0)]
        assert path == ["node1", simple_mol_node]

    def test_hash_consistency(self, simple_mol_node):
        """Test that hash is consistent (uses object id)"""
        hash1 = hash(simple_mol_node)
        hash2 = hash(simple_mol_node)
        assert hash1 == hash2
        assert hash1 == id(simple_mol_node)


class TestRxnNode:
    """Test cases for RxnNode class"""

    @pytest.fixture
    def simple_rxn_node(self):
        """Create a simple RxnNode for testing"""
        return RxnNode(
            smiles="CCO>>CC.O",
            template="[C:1]-[O:2]>>[C:1].[O:2]",
            reagents="H2SO4",
            temp=350.0,
            depth=2,
            cost=[1.0, 2.0],
            weight_length=2,
        )

    def test_initialization_basic(self):
        """Test basic RxnNode initialization"""
        node = RxnNode(
            smiles="CC>>C.C",
            template="[C:1]-[C:2]>>[C:1].[C:2]",
            reagents="heat",
            temp=400.0,
            depth=1,
            cost=[0.5, 1.5],
            weight_length=3,
        )

        assert node.smiles == "CC>>C.C"
        assert node.template == "[C:1]-[C:2]>>[C:1].[C:2]"
        assert node.reagents == "heat"
        assert node.temp == 400.0
        assert node.depth == 1
        assert len(node.cost) == 2
        assert len(node.rxn_no) == 3
        assert len(node.total_value) == 3
        assert not node.success

    def test_initialization_empty_cost(self):
        """Test that empty cost raises ValueError"""
        with pytest.raises(ValueError, match="Reaction cost must be provided"):
            RxnNode(
                smiles="CC>>C.C",
                template="[C:1]-[C:2]>>[C:1].[C:2]",
                reagents="heat",
                temp=400.0,
                depth=1,
                cost=[],
                weight_length=2,
            )

    def test_delta_offset_uniqueness(self):
        """Test that delta offset makes costs unique"""
        node1 = RxnNode(
            smiles="CC>>C.C",
            template="template1",
            reagents="reagent1",
            temp=300.0,
            depth=1,
            cost=[1.0, 2.0],
            weight_length=2,
        )

        node2 = RxnNode(
            smiles="CC>>C.C",
            template="template2",  # Different template
            reagents="reagent1",
            temp=300.0,
            depth=1,
            cost=[1.0, 2.0],
            weight_length=2,
        )

        # Costs should be different due to delta offset
        assert node1.cost != node2.cost
        assert node1.true_cost == node2.true_cost == [1.0, 2.0]

    def test_uppropagate_successful_children(self, simple_rxn_node):
        """Test uppropagation with all successful children"""
        child1 = Mock(spec=MolNode)
        child1.rxn_no = [1.0, 2.0]
        child1.success = True
        child1.success_cost = {(0.5, 1.0): ["mol1"]}

        child2 = Mock(spec=MolNode)
        child2.rxn_no = [2.0, 1.0]
        child2.success = True
        child2.success_cost = {(1.0, 0.5): ["mol2"]}

        weights = np.array([[1.0, 0.0], [0.0, 1.0]])

        result = simple_rxn_node.uppropagate([child1, child2], weights)

        assert result
        assert simple_rxn_node.success
        # Expected: child1 + child2 + reaction cost
        # [1.0, 2.0] + [2.0, 1.0] + weights @ [1.0, 2.0] = [3.0, 3.0] + [1.0, 2.0] = [4.0, 5.0]
        expected_rxn_no = [4.0, 5.0]
        np.testing.assert_array_almost_equal(
            simple_rxn_node.rxn_no, expected_rxn_no, decimal=10
        )

    def test_uppropagate_unsuccessful_children(self, simple_rxn_node):
        """Test uppropagation with unsuccessful children"""
        child1 = Mock(spec=MolNode)
        child1.rxn_no = [1.0, 2.0]
        child1.success = True
        child1.success_cost = {(0.5, 1.0): ["mol1"]}

        child2 = Mock(spec=MolNode)
        child2.rxn_no = [2.0, 1.0]
        child2.success = False  # This child is not successful
        child2.success_cost = {}

        weights = np.array([[1.0, 0.0], [0.0, 1.0]])

        result = simple_rxn_node.uppropagate([child1, child2], weights)

        assert result
        assert not simple_rxn_node.success  # Not all children are successful

    def test_uppropagate_no_children(self, simple_rxn_node):
        """Test that uppropagation with no children raises error"""
        weights = np.array([[1.0, 0.0]])

        with pytest.raises(ValueError, match="No children provided for RxnNode"):
            simple_rxn_node.uppropagate([], weights)

    def test_downpropagate(self, simple_rxn_node):
        """Test downpropagation from parent molecule"""
        parent = Mock(spec=MolNode)
        parent.rxn_no = [3.0, 4.0]
        parent.total_value = [1.0, 1.5]

        simple_rxn_node.rxn_no = [5.0, 6.0]

        result = simple_rxn_node.downpropagate(parent)

        assert result
        # Expected: rxn_no - parent.rxn_no + parent.total_value
        # [5.0, 6.0] - [3.0, 4.0] + [1.0, 1.5] = [3.0, 3.5]
        assert simple_rxn_node.total_value == [3.0, 3.5]

    def test_track_success_cost_simple(self, simple_rxn_node):
        """Test simple success cost tracking"""
        child1 = Mock(spec=MolNode)
        child1.success_cost = {(1.0, 2.0): ["mol1"]}

        child2 = Mock(spec=MolNode)
        child2.success_cost = {(0.5, 1.0): ["mol2"]}

        simple_rxn_node.success_cost = {}

        result = simple_rxn_node.track_success_cost([child1, child2])

        # Expected total cost: (1.0, 2.0) + (0.5, 1.0) + reaction cost
        # Reaction cost includes delta offset, so we check the structure
        assert len(result) == 1
        cost_key = list(result.keys())[0]
        path = result[cost_key]

        assert "mol1" in path
        assert "mol2" in path
        assert simple_rxn_node in path

    def test_track_success_cost_multiple_combinations(self, simple_rxn_node):
        """Test success cost tracking with multiple cost combinations"""
        child1 = Mock(spec=MolNode)
        child1.success_cost = {(1.0, 2.0): ["mol1a"], (2.0, 1.0): ["mol1b"]}

        child2 = Mock(spec=MolNode)
        child2.success_cost = {(0.5, 1.0): ["mol2"]}

        simple_rxn_node.success_cost = {}

        result = simple_rxn_node.track_success_cost([child1, child2])

        # Should have 2 combinations: (1.0,2.0)+(0.5,1.0) and (2.0,1.0)+(0.5,1.0)
        assert len(result) == 2

    def test_track_success_cost_with_tuples(self, simple_rxn_node):
        """Test success cost tracking with tuple format (target paths)"""
        child = Mock(spec=MolNode)
        child.success_cost = {(1.0, 2.0): (["mol1"], {0, 1})}

        simple_rxn_node.success_cost = {}

        result = simple_rxn_node.track_success_cost([child])

        assert len(result) == 1
        cost_key = list(result.keys())[0]
        path = result[cost_key]

        assert "mol1" in path
        assert simple_rxn_node in path

    def test_hash_consistency(self, simple_rxn_node):
        """Test that hash is consistent (uses object id)"""
        hash1 = hash(simple_rxn_node)
        hash2 = hash(simple_rxn_node)
        assert hash1 == hash2
        assert hash1 == id(simple_rxn_node)


class TestUtilityFunctions:
    """Test utility functions"""

    def test_zero_vector(self):
        """Test zero vector creation"""
        result = zero_vector(3)
        assert result == [0.0, 0.0, 0.0]

        result = zero_vector(0)
        assert result == []


class TestIntegration:
    """Integration tests for MolNode and RxnNode interaction"""

    @pytest.fixture
    def heuristic_fns(self):
        def h1(smiles):
            return 1.0

        def h2(smiles):
            return 2.0

        return [h1, h2]

    def test_mol_rxn_propagation_cycle(self, heuristic_fns):
        """Test a complete propagation cycle between MolNode and RxnNode"""
        # Create a known starting material
        bb_node = MolNode(
            smiles="CC", heuristic_fns=heuristic_fns, depth=2, is_known=True
        )  # Create a reaction that uses this building block
        rxn_node = RxnNode(
            smiles="CCO>>CC.O",  # Use valid reaction SMILES
            template="template",
            reagents="reagent",
            temp=300.0,
            depth=1,
            cost=[0.5, 1.0],
            weight_length=2,
        )

        # Create target molecule
        target_node = MolNode(
            smiles="CCO",  # Use valid SMILES
            heuristic_fns=heuristic_fns,
            depth=0,
            is_known=False,
            is_target=True,
        )
        weights = np.array([[1.0, 0.0], [0.0, 1.0]])

        # Simulate uppropagation from building block
        bb_updated = bb_node.uppropagate([], weights)
        assert bb_updated
        assert bb_node.success

        # Propagate to reaction
        rxn_updated = rxn_node.uppropagate([bb_node], weights)
        assert rxn_updated
        assert rxn_node.success

        # Mark target as not open since it has children (reaction)
        target_node.is_open = False

        # Propagate to target
        target_updated = target_node.uppropagate([rxn_node], weights)
        assert target_updated
        assert target_node.success

        # Check that costs are properly tracked
        assert len(target_node.success_cost) > 0


class TestEdgeCases:
    """Test edge cases and critical scenarios"""

    @pytest.fixture
    def heuristic_fns(self):
        def h1(smiles):
            return 1.0

        def h2(smiles):
            return 2.0

        return [h1, h2]

    def test_mol_node_no_change_uppropagate(self, heuristic_fns):
        """Test uppropagate when no actual changes occur"""
        node = MolNode(
            smiles="CCO", heuristic_fns=heuristic_fns, depth=1, is_known=False
        )

        # Set initial state
        node.rxn_no = [1.5]
        node.success = False

        weights = np.array([[0.5, 0.5]])  # Will produce same rxn_no = [1.5]

        result = node.uppropagate([], weights)
        assert not result  # No changes should occur

    def test_rxn_node_duplicate_uppropagate_calls(self):
        """Test that duplicate RxnNode uppropagate calls behave correctly"""
        rxn_node = RxnNode(
            smiles="CCO>>CC.O",
            template="template",
            reagents="reagent",
            temp=300.0,
            depth=1,
            cost=[1.0, 2.0],
            weight_length=2,
        )

        # Create mock children
        child = Mock(spec=MolNode)
        child.rxn_no = [1.0, 2.0]
        child.success = True
        child.success_cost = {(0.5, 1.0): ["mol1"]}

        weights = np.array([[1.0, 0.0], [0.0, 1.0]])

        # First call should update
        result1 = rxn_node.uppropagate([child], weights)
        assert result1

        # Second call with same parameters should not update
        result2 = rxn_node.uppropagate([child], weights)
        assert not result2  # No changes

    def test_mol_node_downpropagate_no_change(self, heuristic_fns):
        """Test downpropagate when no changes occur"""
        node = MolNode(
            smiles="CCO", heuristic_fns=heuristic_fns, depth=1, is_known=False
        )

        # Set initial total_value
        node.total_value = [1.0, 2.0]

        # Create parent with same total_value
        parent = Mock(spec=RxnNode)
        parent.total_value = [1.0, 2.0]

        result = node.downpropagate([parent])
        assert not result  # No change should occur

    def test_rxn_node_downpropagate_no_change(self):
        """Test RxnNode downpropagate when no changes occur"""
        rxn_node = RxnNode(
            smiles="CCO>>CC.O",
            template="template",
            reagents="reagent",
            temp=300.0,
            depth=1,
            cost=[1.0, 2.0],
            weight_length=2,
        )

        # Set up state that would result in no change
        rxn_node.rxn_no = [5.0, 6.0]
        rxn_node.total_value = [3.0, 3.5]

        parent = Mock(spec=MolNode)
        parent.rxn_no = [3.0, 4.0]
        parent.total_value = [1.0, 1.5]

        # This should result in same total_value: [5.0, 6.0] - [3.0, 4.0] + [1.0, 1.5] = [3.0, 3.5]
        result = rxn_node.downpropagate(parent)
        assert not result  # No change

    def test_track_success_cost_existing_costs(self, heuristic_fns):
        """Test that existing costs in success_cost are not overwritten"""

        node = MolNode(
            smiles="CCO", heuristic_fns=heuristic_fns, depth=1, is_known=False
        )

        # Pre-populate success_cost with mock node
        existing_node = Mock(spec=MolNode)
        node.success_cost = {(1.0, 2.0): [existing_node]}

        # Create child with same cost vector
        child = Mock(spec=RxnNode)
        new_node = Mock(spec=MolNode)
        child.success_cost = {(1.0, 2.0): [new_node]}

        result = node.track_success_cost([child])

        # Should not add duplicate cost
        assert len(result) == 0
        assert node.success_cost[(1.0, 2.0)] == [existing_node]  # Unchanged

    def test_success_cost_with_empty_successors(self, heuristic_fns):
        """Test success cost tracking with empty successors"""
        node = MolNode(
            smiles="CCO", heuristic_fns=heuristic_fns, depth=1, is_known=False
        )

        # Create child with empty successor
        child = Mock(spec=RxnNode)
        child.success_cost = {(1.0, 2.0): []}  # Empty successor list

        result = node.track_success_cost([child])

        # Should filter out empty successors
        assert len(result) == 0

    def test_rxn_node_assert_error_empty_rxn_no(self):
        """Test assertion error when child has empty rxn_no"""
        rxn_node = RxnNode(
            smiles="CCO>>CC.O",
            template="template",
            reagents="reagent",
            temp=300.0,
            depth=1,
            cost=[1.0, 2.0],
            weight_length=2,
        )

        # Create child with empty rxn_no
        child = Mock(spec=MolNode)
        child.rxn_no = []  # Empty rxn_no
        child.success = True
        child.success_cost = {}

        weights = np.array([[1.0, 0.0]])

        with pytest.raises(
            AssertionError, match="Rxn_no for MolNode should not be empty"
        ):
            rxn_node.uppropagate([child], weights)
