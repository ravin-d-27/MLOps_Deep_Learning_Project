import logging
from typing import Tuple
from typing_extensions import Annotated
import pandas as pd
from zenml import step
from src.Data_Cleaning import DataPreprocessing, DataSplitting


@step
def clean_data(data: Annotated[pd.DataFrame, "Data which needs to be cleaned"]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Clean the given data by performing necessary preprocessing steps.

    Args:
        data (pd.DataFrame): Data which needs to be cleaned.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing the cleaned data (X) and the corresponding labels (y).
    """
    
    try:
        logging.info("***** Data Preprocessing started in clean_data.py *****")
        data_clean = DataPreprocessing()
        X,y = data_clean.clean_data(data)
        logging.info("***** Data Preprocessing in clean_data.py is successfully executed *****")
        return X,y
        
    except Exception as e:
        logging.error("### Failed to perform Data Preprocessing in clean_data.py: {} ###".format(e))
        raise e


@step
def split_data(X: Annotated[pd.DataFrame, "Features"], 
               y: Annotated[pd.Series, "Independent Variables"]) -> Tuple[pd.DataFrame, 
                                                                             pd.DataFrame, 
                                                                             pd.Series, 
                                                                             pd.Series]:
    """
    Split the data into training and testing sets.

    Args:
        X (pd.DataFrame): Features.
        y (pd.DataFrame): Independent Variables.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing the training features (X_train),
        testing features (X_test), training labels (y_train), and testing labels (y_test).
    """

    try:
        logging.info("***** Data Splitting started in clean_data.py *****")
        data_split = DataSplitting()
        X_train, X_test, y_train, y_test = data_split.clean_data(X,y)
        logging.info("***** Data Splitting in clean_data.py is successfully executed *****")
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logging.error("### Failed to perform Data Splitting clean_data.py: {} ###".format(e))
        raise e
    
    
    