from abc import ABC, abstractmethod
from typing import List
import json
from ..processor import Step,Pipeline

class BaseExtractor(ABC):
    def __init__(self, config_file_path: str):
        """
        Initializes the BaseExtractor with a configuration file path.

        Args:
            config_file_path (str): Path to the configuration file.
        """
        self.config_file_path = config_file_path
        self.config = self.load_config()

    def load_config(self):
        """
        Loads the configuration file from the specified path.

        Returns:
            dict: The loaded configuration as a dictionary.
        """
        with open(self.config_file_path, 'r') as f:
            return json.load(f)

    @abstractmethod
    def extract_steps(self) -> List['Step']:
        """
        Abstract method that must be implemented by subclasses to define the extraction steps.

        Returns:
            List[Step]: A list of Step objects representing the extraction steps.
        """
        pass

    @abstractmethod
    def make_pipeline(self)->Pipeline:
        """
        Abstract method that must be implemented by subclasses to create a specific pipeline.

        Returns:
            Pipeline: The created pipeline object.
        """
        pass

