"""
This module defines the Models class, which is used to train and select the best machine learning model 
for regression tasks using GridSearchCV. The class supports Support Vector Regressor (SVR) and 
Gradient Boosting Regressor (GRB) with predefined hyperparameters.

Classes:
    Models: A class that initializes regressors and their hyperparameters, and performs grid search 
            to find the best model.

Methods:
    __init__(self):
        Initializes the Models class with dictionaries of regressors and their hyperparameters.

    grid_training(self, X, y):
        Performs grid search with cross-validation to find the best model based on the provided 
        training data and exports the best model using the Utils class.

        Parameters:
            X (pd.DataFrame): The input features for training.
            y (pd.Series): The target variable for training.
"""
import pandas as pd
import numpy as np

from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from utils import Utils

class Models:
    def __init__(self):
        # Initialize the dictionary of regressors
        self.reg = {
            'SVR': SVR(),  # Support Vector Regressor
            'GRB': GradientBoostingRegressor()  # Gradient Boosting Regressor
        }

        # Initialize the dictionary of hyperparameters for each regressor
        self.params = {
            'SVR': {
                'kernel': ['linear', 'poly', 'rbf'],  # Kernel type to be used in the algorithm
                'gamma': ['scale', 'auto'],  # Kernel coefficient
                'C': [1, 10, 100]  # Regularization parameter
            },
            'GRB': {
                'loss': ['squared_error', 'absolute_error'],  # Loss function to be optimized
                'learning_rate': [0.01, 0.05, 0.1]  # Learning rate shrinks the contribution of each tree
            }
        }
    def grid_training(self, X, y):
        # Initialize variables to store the best score and best model
        best_score = 999
        best_model = None

        # Iterate over each regressor in the dictionary
        for name, reg in self.reg.items():
            # Perform grid search with cross-validation
            grid_reg = GridSearchCV(reg, self.params[name], cv=3)
            grid_reg.fit(X, y.values.ravel())
            # Get the absolute value of the best score
            score = np.abs(grid_reg.best_score_)

            # Update the best score and best model if the current score is better
            if score < best_score:
                best_score = score
                best_model = grid_reg.best_estimator_

        # Export the best model using the Utils class
        utils = Utils() 
        utils.model_export(best_model, best_score)