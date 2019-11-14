#!/usr/bin/env python3

import numpy as np


class RandomRecommender:
    def __init__(self):
        self.numItems = None

    def __str__(self):
        return 'Random Recommender'

    def fit(self, urm_train):
        self.numItems = urm_train.shape[0]

    def recommend(self, user_id, at=5):
        return np.random.choice(self.numItems, at)