import logging
from zenml import step
import pandas as pd

import tensorflow as tf
from abc import ABC, abstractmethod

class DL_Model(ABC):
    
    """Abstract class for Deep Learning Models"""
    
    @abstractmethod
    def train(self):
        pass
    

class Artificial_Neural_Networks(DL_Model):
    
    def train(self, X_train: pd.DataFrame , y_train: pd.Series):
        
        try:
            logging.info("*** DL Model Training Started ***")
            ann = tf.keras.Sequential()
            ann.add(tf.keras.layers.Dense(50, "relu"))
            ann.add(tf.keras.layers.Dense(25, "relu"))
            ann.add(tf.keras.layers.Dense(12, "relu"))
            ann.add(tf.keras.layers.Dense(units=8, activation=tf.keras.layers.Softmax()))
            ann.compile(optimizer='adam',loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
            
            ann.fit(X_train, y_train, epochs=10, batch_size=128)
            
            return ann
            
        except Exception as e:
            logging.error("DL Model Training Failed: {}".format(e))