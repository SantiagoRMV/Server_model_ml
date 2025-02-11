import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.svm import SVR
from sklearn.linear_model import RANSACRegressor, HuberRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    
    warnings.simplefilter("ignore")

    # Dataframes
    df = pd.read_csv('in/felicidad_corrupt.csv')
    X = df[['gdp', 'family', 'lifexp', 'freedom', 'corruption', 'generosity', 'dystopia']]
    y = df[['score']]

    # Split train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Model train
    estimadores = {
        'SVR': SVR(gamma='auto', C=1.0, epsilon=0.1),
        'RANSAC': RANSACRegressor(),
        'Huber': HuberRegressor(epsilon=1.35) # Param by default
    }

    for key, estimator in estimadores.items():
        estimator.fit(X_train, y_train)
        y_predictions = estimator.predict(X_test)
        model_loss = mean_squared_error(y_test, y_predictions)
        
        print('-'*60)
        print(f'MSE {key}:{model_loss}')