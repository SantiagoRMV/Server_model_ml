import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    # Data frames
    dt_heart = pd.read_csv("in/heart.csv")
    X = dt_heart.drop(['target'], axis=1)
    y = dt_heart['target']

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Model
    boost = GradientBoostingClassifier(n_estimators=50).fit(X_train, y_train) # Estimators are equal to trees
    boost_pred = boost.predict(X_test)
    print('-'*60)
    print(f'Acuracy score: {accuracy_score(boost_pred, y_test)}')