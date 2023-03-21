
from src.components.data_ingestion import DataIngestionConfig
from src.components.data_ingestion import DataIngestion

from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation

from src.components.model_selection import ModelSelectionConfig
from src.components.model_selection import ModelSelection

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

from src.components.model_testing import ModelTestingConfig
from src.components.model_testing import ModelTesting

from src.components.best_model import BestModelConfig
from src.components.best_model import BestModel


if __name__=="__main__":
    Data_Ingestion=DataIngestion()
    train_data,test_data=Data_Ingestion.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr=data_transformation.initiate_data_transformation(train_data,test_data)

    model_selction=ModelSelection()
    models, params=model_selction.initiate_model_selection()

    model_training=ModelTrainer()
    model=model_training.initiate_model_training(train_arr, test_arr, models, params)

    model_testing=ModelTesting()
    report=model_testing.Initiate_model_testing(train_arr, test_arr, models, model)

    best_model=BestModel()
    print(best_model.get_best_model(models,report))