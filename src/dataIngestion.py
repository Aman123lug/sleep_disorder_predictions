import pandas as pd
from loggerr import logger

STAGE_NAME  = "Data Ingestion Stage"

class DataIngestion:
    def __init__(self) -> None:
        pass
    def data_ingestion(self):

        df = pd.read_csv("data/sleep-dataset.csv")
        
        return df
        
if __name__ == "__main__":
    
    try:
        logger.info(f" >>>> stage {STAGE_NAME} <<<< started !")
        obj = DataIngestion()
        data = obj.data_ingestion()
        logger.info(f" >>>> stage {STAGE_NAME} <<<< Completed ! \n\n x==================x")
        
    except Exception as e:
        logger.exception(e)
        raise e
