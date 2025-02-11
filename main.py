"""
This script is the main entry point for training machine learning models using scikit-learn.
It loads data from a CSV file, processes the features and target variables, and performs grid search training.
"""
from utils import Utils
from models import Models

if __name__ == "__main__":
    
    utils = Utils()
    models = Models()

    data = utils.load_from_csv('./in/felicidad.csv')
    X, y = utils.features_target(data, ['country', 'rank', 'score'], ['score'])
    
    models.grid_training(X, y)

