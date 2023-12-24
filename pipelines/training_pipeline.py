from zenml import pipeline

from steps.ingest_data import ingest_data
from steps.clean_data import clean_data, split_data
from steps.model_train import model_train
from steps.model_eval import model_eval

@pipeline
def training_pipeline(path: str):
    """Training pipeline with a single step."""
    
    data = ingest_data(path)
    X,y = clean_data(data)
    
    X_train, X_test, y_train, y_test = split_data(X,y)
    model = model_train(X_train, y_train)
    
    model_eval(X_test, y_test, model)
    