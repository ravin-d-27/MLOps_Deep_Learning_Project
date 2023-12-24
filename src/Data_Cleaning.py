import logging
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Tuple
from typing_extensions import Annotated

from sklearn.model_selection import train_test_split

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
        
    def handle_data(self, data: Annotated[pd.DataFrame, "The input data to be cleaned."]) -> pd.DataFrame:
        """
        This method is used to clean the data and returns the dataframe
        """
        
        try:
            data = data.drop(['DATE_DIED','CLASIFFICATION_FINAL'], axis=1)
            return data
        except Exception as e:
            logging.error("Data Cleaning Failed: {}".format(e))
            raise e
        
        
        
class DataSplitting(DataCleaningStrategy):
    """
    This class provides methods for cleaning data.
    """
    def clean_data(self, X: Annotated[pd.DataFrame, "Features"], 
                   y: Annotated[pd.Series, "Target Variable"]) -> Annotated[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], "Training and Test Set"]:
        """
        Splits the features and Target Variables 

        Parameters:
        - X: Features
        - y: Target

        Returns:
        - X_train, X_test, y_train, y_test
        
        """
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=0)
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logging.error("Data Splitting Failed: {}".format(e))
            raise e