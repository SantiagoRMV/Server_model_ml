import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, KFold

if __name__ == "__main__":
    # Dataset
    dataset = pd.read_csv("in/felicidad.csv")
    X = dataset.drop(['country', 'score'], axis=1)
    y = dataset['score']

    # Modelo
    model = DecisionTreeRegressor()
    score = cross_val_score(model, X, y, scoring='neg_mean_squared_error')
    print(score)
    print(np.abs(np.mean(score)))

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for train, test in kf.split(dataset):
        print(train)
        print(test)
        