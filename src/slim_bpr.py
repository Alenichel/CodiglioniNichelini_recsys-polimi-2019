#!/usr/bin/env python3

from time import time
from datetime import timedelta
import numpy as np
from scipy.special import expit
from Base.Recommender_utils import similarityMatrixTopK


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

    def fit(self, urm_train, epochs=100):
        self.urm_train = urm_train.tocsr()
        self.n_users = urm_train.shape[0]
        self.n_items = urm_train.shape[1]
        # Initialize similarity with random values and zero-out diagonal
        #self.S = np.random.random((self.n_items, self.n_items)).astype('float32')
        #self.S[np.arange(self.n_items), np.arange(self.n_items)] = 0
        self.S = np.zeros((self.n_items, self.n_items), dtype=float)
        start_time_train = time()
        for currentEpoch in range(epochs):
            start_time_epoch = time()
            self.epoch_iteration()
            elapsed_time = timedelta(seconds=int(time() - start_time_epoch))
            print("Epoch {0} of {1} complete in {2}.".format(currentEpoch+1, epochs, elapsed_time))
        elapsed_time = timedelta(seconds=int(time() - start_time_train))
        print("Train completed in {0}.".format(elapsed_time))
        # The similarity matrix is learnt row-wise
        # To be used in the product URM*S must be transposed to be column-wise
        self.W = self.S.T
        del self.S
        # TODO: Check
        self.W = similarityMatrixTopK(self.W, verbose=True).tocsr()
        self.W.eliminate_zeros()

    def epoch_iteration(self):
        num_positive_interactions = int(self.urm_train.nnz * 0.01)
        start_time = time()
        batch_size = 10000
        start_time_batch = start_time
        for num_sample in range(num_positive_interactions):
            user_id, pos_item_id, neg_item_id = self.sample_triple()
            self.update_factors(user_id, pos_item_id, neg_item_id)
            if num_sample % batch_size == 0 and num_sample > 0:
                elapsed = timedelta(seconds=int(time()-start_time))
                samples_ps = batch_size / (time() - start_time_batch)
                eta = timedelta(seconds=int((num_positive_interactions - num_sample) / samples_ps))
                print('Processed {0:7.0f} samples ( {1:5.2f}% ) in {2} | Samples/s: {3:4.0f} | ETA: {4}'.format(
                    num_sample,
                    100.0 * float(num_sample)/num_positive_interactions,
                    elapsed,
                    samples_ps,
                    eta))
                start_time_batch = time()

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
        scores = user_profile.dot(self.W)
        scores = scores.toarray().ravel()
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