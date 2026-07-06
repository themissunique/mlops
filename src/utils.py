#all common functionalities for project

from xml.parsers.expat import model

import dill
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        
        for i in range(len(models)):
            model = list(models.values())[i]
            param_grid = params[list(models.keys())[i]]

            gs = GridSearchCV(model, param_grid, cv=3)
            gs.fit(X_train,y_train)

            # Train model
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            # Predict testing data
            y_test_pred = model.predict(X_test)

            # Get r2 score for test data
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)