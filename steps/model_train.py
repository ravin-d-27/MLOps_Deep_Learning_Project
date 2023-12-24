import logging
import pandas as pd
import tensorflow as tf
from src.Model_Training import Artificial_Neural_Networks
from zenml import step

@step
def model_train(X_train:pd.DataFrame, y_train:pd.DataFrame):
    """ZenML Step for Training the DL Model"""
    try:
        logging.info("*** DL Model Training Started ***")
        ann = Artificial_Neural_Networks()
        model = ann.train(X_train, y_train)
        logging.info("*** DL Model Training Completed ***")
        return model
    except Exception as e:
        logging.error("DL Model Training Failed: {}".format(e))
        raise e