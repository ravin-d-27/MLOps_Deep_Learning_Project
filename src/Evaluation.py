import logging
import pandas as pd
import tensorflow as tf
from abc import ABC, abstractmethod
import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class EvaluationStrategy(ABC):
    
    """Abstract class for Evaluation Strategy"""
    
    @abstractmethod
    def evaluate(self):
        pass
    

class Evaluate(EvaluationStrategy):
    
    def evaluate(self, y_test: pd.Series, y_pred: np.ndarray):
        
        try:
            logging.info("*** DL Model Evaluation Started ***")

            r = []
            for i in range(len(y_pred)):
                r.append(np.argmax(y_pred[i]))
            
            accuracy = accuracy_score(y_test, r)
            logging.info("*** DL Model Evaluation Completed ***")
            
            return accuracy
            
        except Exception as e:
            logging.error("DL Model Evaluation Failed: {}".format(e))
            raise e