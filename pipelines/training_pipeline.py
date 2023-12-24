from zenml import pipeline

from steps.ingest_data import ingest_data
from steps.clean_data import clean_data

@pipeline
def training_pipeline(path: str):
    """Training pipeline with a single step."""
    
    data = ingest_data(path)
    X,y = clean_data(data)