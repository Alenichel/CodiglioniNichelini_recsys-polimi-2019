#!/usr/bin/env python3

import numpy as np
import scipy.sparse as sps

from run import train_test_split
from load_data import load_csv
from export_data import export_csv
from evaluation import evaluate_algorithm
from random_recommender import RandomRecommender


if __name__ == '__main__':
    data = load_csv('old_competition_data/train.csv') + load_csv('old_competition_data/train_sequential.csv')
    data = [[int(x) for x in line] for line in data]
    target_playlists = [int(x[0]) for x in load_csv('old_competition_data/target_playlists.csv')]
    playlists, tracks = map(np.array, zip(*data))
    ratings = np.ones(playlists.size)                                                                            # it creates an array of ones
    urm_train, urm_test = train_test_split(playlists, tracks, ratings, playlists.size)
    recommender = RandomRecommender()
    recommender.fit(urm_train)
    evaluate_algorithm(urm_test, recommender)
    data = [(p_id, recommender.recommend(p_id, at=10)) for p_id in target_playlists]
    export_csv(data)