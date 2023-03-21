import os
import sys
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score


from src.exception import CustomException
from src.logger import logging
from src.utils import model_report_csv
from dataclasses import dataclass


@dataclass
class ModelTestingConfig:
    testing_report_file_path=os.path.join("artifacts","Testing_Evaluation_Report.csv")

class ModelTesting:
    def __init__(self):
        self.model_Testing_config=ModelTestingConfig()

    def Initiate_model_testing(self,train_array, test_array, models, model):
        try:

            logging.info("Split test data")            
            X_train,y_train=(train_array[:,:-1], train_array[:,-1])
            X_test,y_test=(test_array[:,:-1],test_array[:,-1])

            
            
            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report = {}        

            for i in range(len(list(models))):
                report[list(models.keys())[i]] = test_model_score

            logging.info(f"Saved Testing_Evaluation_Report.")

            model_report_csv(
                self.model_Testing_config.testing_report_file_path,
                report
            )


            return report
        
        
        except Exception as e:
            raise CustomException(e, sys)
    


    