#!/usr/bin/env python3

import numpy as np

# It removes biases of global phenomena
class GlobalEffectsRecommender:

    def __init__(self):
        self.urm_train = None
        self.best_rated_items = None

    def __str__(self):
        return 'Global Effects Recommender'

    def fit(self, urm_train):
        self.urm_train = urm_train
        global_average = np.mean(urm_train.data)
        urm_train_unbiased = urm_train.copy()
        urm_train_unbiased.data -= global_average
        # User bias
        user_mean_rating = urm_train_unbiased.mean(axis=1)
        user_mean_rating = np.array(user_mean_rating).squeeze()
        for user_id in range(len(user_mean_rating)):
            start_position = urm_train_unbiased.indptr[user_id]
            end_position = urm_train.indptr[user_id+1]
            urm_train_unbiased.data[start_position:end_position] -= user_mean_rating[user_id]
        # Item bias
        item_mean_rating = urm_train_unbiased.mean(axis=0)
        item_mean_rating = np.array(item_mean_rating).squeeze()
        self.best_rated_items = np.argsort(item_mean_rating)
        self.best_rated_items = np.flip(self.best_rated_items, axis=0)

    def recommend(self, user_id, at=5, remove_seen=True):
        if remove_seen:
            unseen_items_mask = np.in1d(self.best_rated_items, self.urm_train[user_id].indices,
                                        assume_unique=True, invert=True)
            unseen_items = self.best_rated_items[unseen_items_mask]
            recommended_items = unseen_items[0:at]
        else:
            recommended_items = self.best_rated_items[0:at]
        return recommended_items