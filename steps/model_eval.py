from src.Evaluation import Evaluate
from zenml import step
import logging
import pandas as pd
import tensorflow as tf

@step
def model_eval(X_test:pd.DataFrame, y_true: pd.DataFrame, model: tf.keras.Model):
    """This step evaluates the performance of a model.
        Args:
            X_test: Pandas DataFrame.
            y_true: Pandas DataFrame.
            model: Trained model.
        Returns:
            None
    """
    
    try:
        logging.info("Evaluating model...")
        
        y_pred = model.predict(X_test)
        
        acc = Evaluate()
        accuracy_score = acc.evaluate(y_true, y_pred)
        
        logging.info("Evaluation complete!")
        logging.info("Accuracy Score: {}".format(acc))
        
        return accuracy_score
        
    except Exception as e:
        logging.error("Error while evaluating model: {}".format(e))
        raise e