import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common import read_yaml

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self,config):
        self.config = config["data_ingestion"]
        self.local_data_path = self.config["local_data_path"]
        self.train_test_ratio = self.config["train_ratio"]

        os.makedirs(RAW_DIR , exist_ok=True)

        logger.info(f"Data Ingestion started with local file: {self.local_data_path}")

    def copy_local_file(self):
        try:
            shutil.copy(self.local_data_path, RAW_FILE_PATH)
            logger.info(f"CSV file copied from {self.local_data_path} to {RAW_FILE_PATH}")

        except Exception as e:
            logger.error("Error while copying the csv file")
            raise CustomException("Failed to copy csv file", e)
        
    def split_data(self):
        try:
            logger.info("Starting the splitting process")
            data = pd.read_csv(RAW_FILE_PATH)
            train_data , test_data = train_test_split(data , test_size=1-self.train_test_ratio , random_state=42)

            train_data.to_csv(TRAIN_FILE_PATH)
            test_data.to_csv(TEST_FILE_PATH)

            logger.info(f"Train data saved to {TRAIN_FILE_PATH}")
            logger.info(f"Test data saved to {TEST_FILE_PATH}")
        
        except Exception as e:
            logger.error("Error while splitting data")
            raise CustomException("Failed to split data into training and test sets ", e)
        
    def run(self):

        try:
            logger.info("Starting data ingestion process")

            self.copy_local_file()
            self.split_data()

            logger.info("Data ingestion completed sucesfully")
        
        except CustomException as ce:
            logger.error(f"CustomException : {str(ce)}")
        
        finally:
            logger.info("Data ingestion completed")

if __name__ == "__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()




        

