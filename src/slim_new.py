#!/usr/bin/env python3

import sys
import time
import numpy as np
from scipy.special import expit


class SLIM_BPR:
    """
    This class is a python porting of the BPRSLIM algorithm in MyMediaLite written in C#
    The code is identical with no optimizations
    """

    def __init__(self, lambda_i=0.0025, lambda_j=0.00025, learning_rate=0.05):
        self.urm_train = None
        self.n_users = None
        self.n_items = None
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j
        self.learning_rate = learning_rate
        self.normalize = False
        self.sparse_weights = False
        self.S = None
        self.W = None

    def fit(self, urm_train, epochs=1):
        self.urm_train = urm_train
        self.n_users = urm_train.shape[0]
        self.n_items = urm_train.shape[1]
        # Initialize similarity with random values and zero-out diagonal
        self.S = np.random.random((self.n_items, self.n_items)).astype('float32')
        self.S[np.arange(self.n_items), np.arange(self.n_items)] = 0
        start_time_train = time.time()
        for currentEpoch in range(epochs):
            start_time_epoch = time.time()
            self.epoch_iteration()
            print("Epoch {} of {} complete in {:.2f} minutes".format(currentEpoch+1, epochs, float(time.time()-start_time_epoch)/60))
        print("Train completed in {:.2f} minutes".format(float(time.time()-start_time_train)/60))
        # The similarity matrix is learnt row-wise
        # To be used in the product URM*S must be transposed to be column-wise
        self.W = self.S.T
        del self.S

    def update_factors(self, user_id, pos_item_id, neg_item_id):
        # Calculate current predicted score
        user_seen_items = self.urm_train[user_id].indices
        prediction = 0
        for user_seen_item in user_seen_items:
            prediction += self.S[pos_item_id, user_seen_item] - self.S[neg_item_id, user_seen_item]
        x_uij = prediction
        logistic_function = expit(-x_uij)
        # Update similarities for all items except those sampled
        for user_seen_item in user_seen_items:
            # For positive item is PLUS logistic minus lambda*S
            if pos_item_id != user_seen_item:
                update = logistic_function - self.lambda_i * self.S[pos_item_id, user_seen_item]
                self.S[pos_item_id, user_seen_item] += self.learning_rate * update
            # For positive item is MINUS logistic minus lambda*S
            if neg_item_id != user_seen_item:
                update = - logistic_function - self.lambda_j * self.S[neg_item_id, user_seen_item]
                self.S[neg_item_id, user_seen_item] += self.learning_rate * update

    def epoch_iteration(self):
        # Get number of available interactions
        num_positive_interactions = self.urm_train.nnz
        start_time = time.time()
        # Uniform user sampling without replacement
        for num_sample in range(num_positive_interactions):
            user_id, pos_item_id, neg_item_id = self.sample_triple()
            self.update_factors(user_id, pos_item_id, neg_item_id)
            if num_sample % 5000 == 0:
                print("Processed {} ( {:.2f}% ) in {:.4f} seconds. Sample per second: {:.0f}".format(
                    num_sample,
                    100.0 * float(num_sample)/num_positive_interactions,
                    time.time()-start_time,
                    5000 / (time.time() - start_time)))
                sys.stderr.flush()
                start_time = time.time()

    def sample_user(self):
        while True:
            user_id = np.random.randint(0, self.n_users)
            num_seen_items = self.urm_train[user_id].nnz
            if 0 < num_seen_items < self.n_items:
                return user_id

    def sample_item_pair(self, user_id):
        user_seen_items = self.urm_train[user_id].indices
        pos_item_id = user_seen_items[np.random.randint(0, len(user_seen_items))]
        while True:
            neg_item_id = np.random.randint(0, self.n_items)
            if neg_item_id not in user_seen_items:
                return pos_item_id, neg_item_id

    def sample_triple(self):
        user_id = self.sample_user()
        pos_item_id, neg_item_id = self.sample_item_pair(user_id)
        return user_id, pos_item_id, neg_item_id

    def recommend(self, user_id, at=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.urm_train[user_id]
        scores = user_profile.dot(self.W).toarray().ravel()
        if exclude_seen:
            scores = self.filter_seen(user_id, scores)
        # rank items
        ranking = scores.argsort()[::-1]
        return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.urm_train.indptr[user_id]
        end_pos = self.urm_train.indptr[user_id + 1]
        user_profile = self.urm_train.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores
