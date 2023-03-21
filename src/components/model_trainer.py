import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","Training_Evaluation_Report.txt")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_training(self, train_array, test_array, models, params):
        try:
            logging.info("Split training data")
            X_train,y_train=(train_array[:,:-1], train_array[:,-1])
            X_test,y_test=(test_array[:,:-1],test_array[:,-1])


            for i in range(len(list(models))):
                model = list(models.values())[i]
                para=params[list(models.keys())[i]]

                gs = GridSearchCV(model,para,cv=3)
                gs.fit(X_train,y_train)

                model.set_params(**gs.best_params_)
                model.fit(X_train,y_train)

            return model

        except Exception as e:
            raise CustomException(e,sys)

       

    
        
        
            
        