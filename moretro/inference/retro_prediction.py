class OneStepModel:
    """
    A one-step model class for retro prediction.
    This class can be called
    """

    def __init__(self, checkpoint_path: str, template_path: str):
        self.cn = checkpoint_path
        self.template_path = template_path

    def predict(self, mol):
        """
        Predict the retro reaction for a given molecule.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class ConditionPrediction:
    """
    Prediction of reaction conditions given the reaction string
    """

    def __init__(self, model_path: str):
        self.model_path = model_path

    def predict(self, reaction_smiles: str):
        """
        Predict the reaction conditions for a given reaction SMILES.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")
