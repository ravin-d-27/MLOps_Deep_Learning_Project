import logging
import pandas as pd
import tensorflow as tf
from src.Model_Training import Artificial_Neural_Networks
from zenml import step
from zenml.client import Client

import mlflow

experiment_tracker = Client().active_stack.experiment_tracker
@step(experiment_tracker=experiment_tracker.name)
def model_train(X_train:pd.DataFrame, y_train:pd.DataFrame):
    """ZenML Step for Training the DL Model"""
    try:
        logging.info("*** DL Model Training Started ***")
        mlflow.tensorflow.autolog()
        ann = Artificial_Neural_Networks()
        model = ann.train(X_train, y_train)
        logging.info("*** DL Model Training Completed ***")
        return model
    except Exception as e:
        logging.error("DL Model Training Failed: {}".format(e))
        raise e