import logging
from zenml import step
import pandas as pd


class IngestData:
    """Class to ingest data from a given path."""
    def __init__(self, path: str):
        self.path = path
        
    def get_data(self):
        df = pd.read_csv(self.path)
        return df
    

@step
def ingest_data(path: str) -> pd.DataFrame:
    """Ingest data from a given path."""
    
    try:
        logging.info("***** Data Ingestion Started *****")
        ingest = IngestData(path)
        logging.info("***** Data Ingestion Finished Successfully *****")
        return ingest.get_data()
    except Exception as e:
        logging.error("Data Ingestion Failed: {}".format(e))