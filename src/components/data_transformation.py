import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dataclasses import dataclass
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass  #@dataclass decorator is used for automatically initialize the special functions such as : __init__(), __repr__(), __hash__(), __eq__() 
class DataTransformationConfig:
    print("Data")
    if not os.path.exists('artifacts'):
        os.makedirs('artifacts')
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl") 

class DataTransformation:
    print("A")
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    print("B")
    def get_data_transformer_object(Self):
        print("C")
        '''
        this function is responsible for data transformation

        '''
        try:
            print("D")
            numerical_cols = ["reading_score","writing_score"]
            categorical_cols = ['gender', 'race_ethnicity', 'parental_level_of_education', 
                                'lunch', 'test_preparation_course']
            
            num_pipeline=Pipeline(
                steps=[
                        ("imputer",SimpleImputer(strategy="median")),
                        ("scaler",StandardScaler(with_mean=False))
                      ]

            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))

                ]

            )

            logging.info(f"Numerical columns: {categorical_cols}")
            logging.info(f"Categorical columns: {numerical_cols}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_cols),
                    ("col_pipeline",cat_pipeline,categorical_cols)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    print("Y")    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            print("Z")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            print("Read train and test data completed")
            print("Obtaining pre processing object")

            preprocessing_object = self.get_data_transformer_object()

            target_column_name = "math score"

            numerical_cols = ["reading_score","writing_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            print(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_object.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            print(f"Saved preprocessing object.")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_object
            )
            print("Preprocessing object saved successfully.")
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    print('Transfomed')
    print("Transformation Successful.")

