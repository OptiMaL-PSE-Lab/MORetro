"""
Simple test for OneStepModel using gin configuration and real checkpoint files.

Tests one specific case: making a prediction with a molecule using the checkpoint paths
from the gin configuration file.
"""

import pytest
import os
import sys
import gin

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from moretro.inference.retro_prediction import OneStepModel


class TestOneStepModelPrediction:
    """Test prediction with real checkpoint using gin config"""

    def _setup_model(self):
        """Helper method to set up the model with gin config"""
        gin.clear_config()

        # Get paths relative to the project root
        test_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(test_dir)  # Go up from tests/ to project root

        # Load the gin configuration
        config_path = os.path.join(
            project_root, "moretro", "configs", "search_config.gin"
        )

        if not os.path.exists(config_path):
            pytest.skip("Config file not found")

        gin.parse_config_file(config_path)

        # Get the model paths from the config (they are relative paths)
        checkpoint_path = os.path.join(
            project_root, "moretro", "models", "model_retro.pt"
        )
        template_path = os.path.join(
            project_root, "moretro", "models", "idx2template_retro.json"
        )

        if not os.path.exists(checkpoint_path):
            pytest.skip("Checkpoint file not found")
        if not os.path.exists(template_path):
            pytest.skip("Template file not found")

        # Create and return the model
        return OneStepModel(gin.REQUIRED)  # type: ignore

    def _verify_predictions(self, all_predictions, target_molecule, top_n):
        """Helper method to verify prediction structure"""
        assert isinstance(all_predictions, list)
        # For single molecule, we get a list with one element (the predictions for that molecule)
        predictions = (
            all_predictions[0] if len(all_predictions) == 1 else all_predictions
        )

        assert isinstance(predictions, list)
        assert len(predictions) <= top_n  # Should respect top_n limit

        # If we got predictions, check their structure
        if len(predictions) > 0:
            pred = predictions[0]
            assert "rxn_smiles" in pred
            assert "reactants" in pred
            assert "template" in pred
            assert "score" in pred

            # Verify reactants is a list of strings
            assert isinstance(pred["reactants"], list)
            assert all(isinstance(r, str) for r in pred["reactants"])

            # Verify reaction SMILES contains the target
            assert target_molecule in pred["rxn_smiles"]

    def test_prediction_with_single_molecule(self):
        """Test making a prediction using checkpoint paths from gin config"""
        model = self._setup_model()

        # Test prediction with a simple molecule
        target_molecule = "C[C@H](c1ccccc1)N1C[C@]2(C(=O)OC(C)(C)C)C=CC[C@@H]2C1=S"
        predictions = model.predict(target_molecule, top_n=5)

        self._verify_predictions(predictions, target_molecule, 5)

    def test_prediction_with_two_molecules(self):
        """Test making predictions for two different target molecules"""
        model = self._setup_model()

        # Test with two different molecules
        target_molecules = [
            "C[C@H](c1ccccc1)N1C[C@]2(C(=O)OC(C)(C)C)C=CC[C@@H]2C1=S",
            "CCc1ccc(C(=O)O)cc1",  # Simple carboxylic acid
        ]

        # Get predictions for both molecules at once
        all_predictions = model.predict(target_molecules, top_n=3)

        # Verify we got predictions for both molecules
        assert len(all_predictions) == 2  # Should have predictions for both molecules

        # Verify each molecule's predictions
        for i, (target_molecule, predictions) in enumerate(
            zip(target_molecules, all_predictions)
        ):
            assert isinstance(predictions, list)
            assert len(predictions) <= 3  # Should respect top_n limit

            # If we got predictions, check their structure
            if len(predictions) > 0:
                pred = predictions[0]
                assert "rxn_smiles" in pred
                assert "reactants" in pred
                assert "template" in pred
                assert "score" in pred

                # Verify reactants is a list of strings
                assert isinstance(pred["reactants"], list)
                assert all(isinstance(r, str) for r in pred["reactants"])

                # Verify reaction SMILES contains the target
                assert target_molecule in pred["rxn_smiles"]


if __name__ == "__main__":
    pytest.main([__file__])
