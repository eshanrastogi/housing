import os
import sys 
sys.path.append('D:/PracticeProject/houseprice') 
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


from src.Components.data_transformation import DataTransformation

## Initiliaze the Data Ingestion Configuration
##(Artifacts are folder name for storing specific data)

@dataclass
class DataIngestionconfig:
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")
    raw_data_path:str=os.path.join("artifacts","raw.csv")


## Create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion methods Starts")
        try:
            df=pd.read_csv("notebooks/data/gemstone.csv") ## .. is given to move to parent folder
            logging.info("Dataset read as pandas Dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("Train test split")
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Data Ingestion is Completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )



        except Exception as e:
            logging.info("Expection occured at Data Ingestion stage")   
            raise CustomException(e,sys)
        

## Run Data Ingestion

if __name__=="__main__":
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation()

