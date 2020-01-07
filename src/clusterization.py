#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from run_utils import DataFiles, build_all_matrices, train_test_split, SplitType, group_struct, get_cold_users


def get_clusters(n_cluster=4, max_iter=300, remove_cold=False):
    age_data = pd.read_csv(DataFiles.UCM_AGE)
    age_data = age_data.drop(['data'], axis=1)
    age_data = age_data.rename(columns={'row': 'user_id', 'col': 'age'})
    region_data = pd.read_csv(DataFiles.UCM_REGION)
    region_data = region_data.drop(['data'], axis=1)
    region_data = region_data.rename(columns={'row': 'user_id', 'col': 'region'})
    data = pd.merge(age_data, region_data, on='user_id')
    urm, _, _, _ = build_all_matrices()
    cold_users = get_cold_users(urm)
    to_remove = list()
    if remove_cold:
        for index, line in data.iterrows():
            if line['user_id'] not in cold_users:
                to_remove += [index]
        data = data.drop(data.index[to_remove])
    X = data.drop(['user_id'], axis=1)
    kmeans = KMeans(n_clusters=n_cluster, init='k-means++', max_iter=max_iter, n_init=10, random_state=42)
    pred_y = kmeans.fit_predict(X)
    clusters = {cluster: list() for cluster in range(max(pred_y) + 1)}
    for i in range(len(data)):
        cluster_id = pred_y[i]
        user_id = data.user_id.iloc[i]
        clusters[cluster_id].append(user_id)
    return clusters

def get_clusters_profile_length(urm_train, n_clusters=10, check=True, stats=False):
    users = np.arange(urm_train.shape[0])
    profile_length = np.ediff1d(urm_train.indptr)
    if check:
        assert len(users) == len(profile_length)
        for user_id in users:
            assert urm_train[user_id].sum() == profile_length[user_id]
    data_len = len(users)
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
    X = profile_length.reshape(-1, 1)
    clusters = kmeans.fit_predict(X)
    user_to_cluster = dict()
    for user_id in users:
        user_to_cluster[user_id] = clusters[user_id]
    cluster_to_users = dict()
    for cluster_id in range(max(clusters) + 1):
        users_in_cluster = list()
        for i in range(data_len):
            if clusters[i] == cluster_id:
                users_in_cluster.append(users[i])
        users_not_in_cluster = np.isin(users, users_in_cluster, invert=True)
        users_not_in_cluster = users[users_not_in_cluster]
        cluster_to_users[cluster_id] = group_struct(in_group=users_in_cluster, not_in_group=users_not_in_cluster)
    if stats:
        for cluster in cluster_to_users:
            print('Cluster:         ', cluster)
            print('Number of users: ', len(cluster_to_users[cluster].in_group))
            profile_length_cluster = profile_length[cluster_to_users[cluster].in_group]
            print('Min profile len: ', min(profile_length_cluster))
            print('Max profile len: ', max(profile_length_cluster))
            print()
    return cluster_to_users, user_to_cluster


if __name__ == '__main__':
    urm, _, _, _ = build_all_matrices()
    urm_train, urm_test = train_test_split(urm, SplitType.PROBABILISTIC)

    #get_clusters_profile_length(urm_train, n_clusters=4, check=False, stats=True, only_cold=True)
    clusters = get_clusters()