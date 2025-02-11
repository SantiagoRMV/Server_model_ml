import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    # Data frames
    df = pd.read_csv("in/felicidad.csv")
    print(df.describe())

    X = df[['gdp', 'family', 'lifexp', 'freedom', 'corruption', 'generosity', 'dystopia']]
    y = df[['score']]

    print(X.shape, y.shape)

    # Split train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Model training and prediction
    modelLinear = LinearRegression().fit(X_train, y_train)
    y_predict_linear = modelLinear.predict(X_test)
    linear_loss = mean_squared_error(y_test, y_predict_linear)
    print(f'MSE modelo lineal:{linear_loss}')
    print(f'Coeficiente linal: {modelLinear.coef_}')
    
    modelLasso = Lasso(alpha=0.02).fit(X_train, y_train)
    y_predict_lasso = modelLasso.predict(X_test)
    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    print(f'MSE modelo lasso:{lasso_loss}')
    print(f'Coeficiente lasso: {modelLasso.coef_}')

    modelRidge = Ridge(alpha=1).fit(X_train, y_train)
    y_predict_ridge = modelRidge.predict(X_test)
    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    print(f'Rigde loss:{ridge_loss}')
    print(f'Coeficiente ridge:{modelRidge.coef_}')

    