import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
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
    knn_class = KNeighborsClassifier().fit(X_train, y_train)
    knn_pred = knn_class.predict(X_test)
    print('-'*60)
    print(f'Acuracy KNN: {accuracy_score(knn_pred, y_test)}')

    bag_class = BaggingClassifier(estimator=KNeighborsClassifier(), n_estimators=50)
    bag_class.fit(X_train, y_train)
    bag_pred = bag_class.predict(X_test)
    print('-'*60)
    print(f'Acuracy Bagging: {accuracy_score(bag_pred, y_test)}')