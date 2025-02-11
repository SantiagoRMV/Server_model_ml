"""
utils.py

This module provides utility functions for data loading, preprocessing, and model exporting.
It includes methods to load data from CSV files, separate features and target variables,
and export trained models along with their scores.

Classes:
    Utils: A class containing utility methods for data handling and model exporting.

Methods:
    load_from_csv(path): Loads data from a CSV file.
    load_from_mysql(): Placeholder method to load data from a MySQL database.
    features_target(dataset, drop_cols, y): Separates features and target variable from a dataset.
    model_export(clf, score): Exports a trained model and its score.

"""
import pandas as pd
import joblib

class Utils:
    
    # Method to load data from a CSV file
    def load_from_csv(self, path):
        return pd.read_csv(path)
    
    # Placeholder method to load data from a MySQL database
    def load_from_mysql(self):
        pass

    # Method to separate features and target variable from a dataset
    def features_target(self, dataset, drop_cols, y):
        X = dataset.drop(drop_cols, axis=1)  # Drop specified columns to get features
        y = dataset[y]  # Get target variable
        return X, y
    
    # Placeholder method to export a trained model and its score
    def model_export(self, clf, score):
        print(score)
        joblib.dump(clf, './models/best_model.pkl')