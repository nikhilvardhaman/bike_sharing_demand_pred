import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.logger import logging

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param,verbose = 0):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            if verbose > 0:
                logging.info(f"Training model {model}")

            para=param[list(models.keys())[i]]
            
            if verbose > 1:
                logging.info(f"Parameters are {para}")
            gs = GridSearchCV(model,para,cv=3,verbose=3)
            gs.fit(X_train,y_train)

            
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)
            
            if verbose > 2:
                logging.info(f"Training score = {train_model_score}")
                logging.info(f"Testing score = {test_model_score}")

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)