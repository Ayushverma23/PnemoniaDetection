from abc import ABC, abstractmethod
from typing import Any

class DiseaseDetector(ABC):
    """
    Abstract base class for disease detection models.
    """

    @abstractmethod
    def load_model(self) -> None:
        """
        Load the model weights and architecture.
        """
        pass

    @abstractmethod
    def predict(self, input_data: Any) -> float:
        """
        Run inference on the input data and return a probability or score.
        """
        pass
