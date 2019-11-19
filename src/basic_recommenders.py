#!/usr/bin/env python3

import numpy as np


class RandomRecommender:
    def __init__(self):
        self.numItems = None

    def __str__(self):
        return 'Random Recommender'

    def fit(self, urm_train):
        self.numItems = urm_train.shape[1]

    def recommend(self, user_id, at=5):
        return np.random.choice(self.numItems, at)


class TopPopRecommender(object):

    def __init__(self):
        self.urm_train = None
        self.popular_items = None

    def __str__(self):
        return 'Top Pop Recommender'

    def fit(self, urm_train):
        self.urm_train = urm_train
        item_popularity = (urm_train > 0).sum(axis=0)
        item_popularity = np.array(item_popularity).squeeze()
        # We are not interested in sorting the popularity value,
        # but to order the items according to it
        self.popular_items = np.argsort(item_popularity)
        self.popular_items = np.flip(self.popular_items, axis=0)

    def recommend(self, user_id, at=5, remove_seen=True):
        if remove_seen:
            unseen_items_mask = np.in1d(self.popular_items, self.urm_train[user_id].indices,
                                        assume_unique=True, invert=True)
            unseen_items = self.popular_items[unseen_items_mask]
            recommended_items = unseen_items[0:at]
        else:
            recommended_items = self.popular_items[0:at]
        return recommended_items
