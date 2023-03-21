
import os
import sys
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score

from src.utils import save_object
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

@dataclass
class BestModelConfig:
    bestmodel_file_path=os.path.join("artifacts","model.pkl")

class BestModel:
    def __init__(self):
        self.best_model_config=BestModelConfig()

    def get_best_model(self,models,report):

        try:
            model_report:dict=report

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model= models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.best_model_config.bestmodel_file_path,
                obj=best_model
            )

           
            return (best_model, best_model_score)
            
        except Exception as e:
            raise CustomException(e,sys)

        

        
    




