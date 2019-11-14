#!/usr/bin/env python3

from load_data import load_csv
from export_data import export_csv
from evaluation import evaluate_algorithm
from random_recommender import RandomRecommender
from global_effects_recommender import GlobalEffectsRecommender
import numpy as np
import scipy.sparse as sps


if __name__ == '__main__':
    data = load_csv('old_competition_data/train.csv') + load_csv('old_competition_data/train_sequential.csv')
    playlists = list()
    tracks = list()
    for line in data:
        playlists.append(line[0])
        tracks.append(line[1])
    playlists = np.array(playlists)
    tracks = np.array(tracks)
    ratings = np.ones(playlists.size)                                                                            # it creates an array of ones
    urm = sps.coo_matrix((ratings, (playlists, tracks)))
    train_test_split = 0.8
    train_mask = np.random.choice([True, False], urm.nnz, p=[train_test_split, 1-train_test_split])             # randomly create the mask
    urm_train = sps.coo_matrix((ratings[train_mask], (playlists[train_mask], tracks[train_mask]))).tocsr()      # we get the training set
    test_mask = np.logical_not(train_mask)
    urm_test = sps.coo_matrix((ratings[test_mask], (playlists[test_mask], tracks[test_mask]))).tocsr()          # we get the test set

    recommender = RandomRecommender()
    recommender.fit(urm_train)
    #evaluate_algorithm(urm_test, recommender)

    data = [(p_id, recommender.recommend(p_id, at=10)) for p_id in range(urm.shape[0])]
    export_csv(data)