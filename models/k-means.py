import pandas as pd

from sklearn.cluster import MiniBatchKMeans

if __name__ == "__main__":
    # Datasets
    dt_candy = pd.read_csv("in/candy.csv")
    X = dt_candy.drop("competitorname", axis=1)

    # Modelo
    k_means = MiniBatchKMeans(n_clusters=4, batch_size=8).fit(X)
    print(f"Total de centroides: {len(k_means.cluster_centers_)}")
    k_predict = k_means.predict(X)
    print(k_predict)

    dt_candy['group'] = k_predict
    print(dt_candy)
