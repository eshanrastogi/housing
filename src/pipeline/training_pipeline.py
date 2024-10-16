import os
import sys
sys.path.append('D:/PracticeProject/houseprice') 
from src.logger import logging
from src.exception import CustomException
import pandas as pd


from src.Components.data_ingestion import DataIngestion
from src.Components.data_transformation import DataTransformation
from src.Components.model_trainer import ModelTrainer










## Run Data Ingestion

if __name__=="__main__":
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initaite_data_transformation(train_data_path,test_data_path)

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_arr , test_arr)