import logging
from typing import Tuple
from typing_extensions import Annotated
from zenml import step
import pandas as pd

from src.Data_Cleaning import DataPreprocessing


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
        logging.error("### Failed to perform clean_data.py: {} ###".format(e))
        raise e
