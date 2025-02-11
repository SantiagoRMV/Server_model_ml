import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    # Dataset
    dataset = pd.read_csv("in/felicidad.csv")
    X = dataset.drop(['country', 'rank', 'score'], axis=1)
    y = dataset['score']

    # Modelo
    reg = RandomForestRegressor()
    parametros = {
        'n_estimators': range(4,16), # Number of trees in the forest
        'criterion': ['absolute_error', 'squared_error'], # Function to measure the quality of a split
        'max_depth': range(2,11) # Maximum depth of the tree
    }

    rand_est = RandomizedSearchCV(reg, parametros, n_iter=10, cv=3, scoring='neg_mean_absolute_error').fit(X, y)
    print(rand_est.best_params_)
    print('-'*60)
    print(rand_est.best_estimator_)

    print(rand_est.predict(X.loc[[0]])) # Predict the score of the first country