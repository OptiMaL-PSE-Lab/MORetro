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

    def test_prediction_with_gin_config_paths(self):
        """Test making a prediction using checkpoint paths from gin config"""
        gin.clear_config()
        
        # Get paths relative to the project root
        test_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(test_dir)  # Go up from tests/ to project root
        
        # Load the gin configuration
        config_path = os.path.join(project_root, "moretro", "configs", "search_config.gin")
        
        if not os.path.exists(config_path):
            pytest.skip("Config file not found")
            
        gin.parse_config_file(config_path)
        
        # Get the model paths from the config (they are relative paths)
        checkpoint_path = os.path.join(project_root, "moretro", "models", "model_retro.pt")
        template_path = os.path.join(project_root, "moretro", "models", "idx2template_retro.json")
        
        if not os.path.exists(checkpoint_path):
            pytest.skip("Checkpoint file not found")
        if not os.path.exists(template_path):
            pytest.skip("Template file not found")
            
        # Create the model
        model = OneStepModel(gin.REQUIRED) # type: ignore
        
        # Test prediction with a simple molecule
        target_molecule = "C[C@H](c1ccccc1)N1C[C@]2(C(=O)OC(C)(C)C)C=CC[C@@H]2C1=S"
        predictions = model.predict(target_molecule, top_n=5)
        
        # Verify the prediction results
        assert isinstance(predictions, list)
        assert len(predictions) <= 5  # Should respect top_n limit
        
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
