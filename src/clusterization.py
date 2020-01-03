#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from run_utils import DataFiles


def get_clusters():
    np.random.seed(42)
    age_data = pd.read_csv(DataFiles.UCM_AGE)
    age_data = age_data.drop(['data'], axis=1)
    age_data = age_data.rename(columns={'row': 'user_id', 'col': 'age'})
    region_data = pd.read_csv(DataFiles.UCM_REGION)
    region_data = region_data.drop(['data'], axis=1)
    region_data = region_data.rename(columns={'row': 'user_id', 'col': 'region'})
    data = pd.merge(age_data, region_data, on='user_id')
    X = data.drop(['user_id'], axis=1)
    kmeans = KMeans(n_clusters=10, init='k-means++', max_iter=300, n_init=10, random_state=42)
    pred_y = kmeans.fit_predict(X)
    clusters = {cluster: list() for cluster in range(max(pred_y) + 1)}
    for i in range(len(data)):
        cluster_id = pred_y[i]
        user_id = data.user_id.iloc[i]
        clusters[cluster_id].append(user_id)
    return clusters
if __name__ == '__main__':
    np.random.seed(42)
    age_data = pd.read_csv(DataFiles.UCM_AGE)
    age_data = age_data.drop(['data'], axis=1)
    age_data = age_data.rename(columns={'row': 'user_id', 'col': 'age'})
    region_data = pd.read_csv(DataFiles.UCM_REGION)
    region_data = region_data.drop(['data'], axis=1)
    region_data = region_data.rename(columns={'row': 'user_id', 'col': 'region'})
    data = pd.merge(age_data, region_data, on='user_id')
    X = data.drop(['user_id'], axis=1)
    '''wcss = []
    for i in range(1, 21):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 21), wcss)
    plt.show()'''
    kmeans = KMeans(n_clusters=10, init='k-means++', max_iter=300, n_init=10, random_state=42)
    pred_y = kmeans.fit_predict(X)
    plt.scatter(data['age'], data['region'])
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
    plt.show()
    clusters = {cluster: list() for cluster in range(max(pred_y) + 1)}
    for i in range(len(data)):
        cluster_id = pred_y[i]
        user_id = data.user_id.iloc[i]
        clusters[cluster_id].append(user_id)
    '''kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10)
    pred_y = kmeans.fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
    plt.show()
    count = 0
    for i in range(len(y)):
        if y[i] == pred_y[i]:
            count += 1
    print(count)'''
