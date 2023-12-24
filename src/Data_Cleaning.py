import logging
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Tuple
from typing_extensions import Annotated

class DataCleaningStrategy(ABC):
    """
    Abstract base class for data cleaning strategies.
    
    This class provides a blueprint for implementing different data cleaning strategies.
    Subclasses of this class should override the `clean_data` method to define their specific data cleaning logic.
    
    """
    
    @abstractmethod
    def clean_data(self):
        """abstract method of DataCleaningStrategy class"""
        pass
    

class DataPreprocessing(DataCleaningStrategy):
    """
    This class provides methods for cleaning data.
    """
    def clean_data(self, data: Annotated[pd.DataFrame, "The input data to be cleaned."]) -> Annotated[Tuple[pd.DataFrame, pd.Series], "The cleaned data."]:
        """
        Cleans the given data and gives features and Target Variables

        Parameters:
        - data: The input data to be cleaned.

        Returns:
        - The cleaned data.
        """
        
        try:
            data = data.drop(['DATE_DIED'], axis=1)
            X = data.drop(['CLASIFFICATION_FINAL'], axis=1)
            y = data['CLASIFFICATION_FINAL']
            return X, y
        except Exception as e:
            logging.error("Data Cleaning Failed: {}".format(e))
            raise e
        
        
        