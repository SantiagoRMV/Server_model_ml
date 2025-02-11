import pandas as pd

from sklearn.cluster import MeanShift

if __name__ == "__main__":
    # Dataset
    dataset = pd.read_csv("in/candy.csv")
    X = dataset.drop("competitorname", axis=1)

    # Modelo
    meanshift = MeanShift().fit(X)
    print(f'Total de centroides: {len(meanshift.cluster_centers_)}')
    y = meanshift.predict(X)
    print(y)

    dataset['grupo'] = y