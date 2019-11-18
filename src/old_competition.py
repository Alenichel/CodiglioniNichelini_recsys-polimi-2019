#!/usr/bin/env python3

import numpy as np
import scipy.sparse as sps

from load_export_csv import load_csv, export_csv
from evaluation import evaluate_algorithm
from random_recommender import RandomRecommender


if __name__ == '__main__':
    data = load_csv('old_competition_data/train.csv') + load_csv('old_competition_data/train_sequential.csv')
    target_playlists = [int(x[0]) for x in load_csv('old_competition_data/target_playlists.csv')]
    playlists, tracks = map(np.array, zip(*data))
    ratings = np.ones(playlists.size)                                                                            # it creates an array of ones
    split = 0.8
    train_mask = np.random.choice([True, False], playlists.size, p=[split, 1-split])             # randomly create the mask
    urm_train = sps.csr_matrix((ratings[train_mask], (playlists[train_mask], tracks[train_mask])))      # we get the training set
    test_mask = np.logical_not(train_mask)
    urm_test = sps.csr_matrix((ratings[test_mask], (playlists[test_mask], tracks[test_mask])))          # we get the test set
    recommender = RandomRecommender()
    recommender.fit(urm_train)
    evaluate_algorithm(urm_test, recommender)
    data = [(p_id, recommender.recommend(p_id, at=10)) for p_id in target_playlists]
    export_csv(('playlist_id', 'tracks_id'), data)