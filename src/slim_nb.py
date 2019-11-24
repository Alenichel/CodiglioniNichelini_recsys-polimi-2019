#!/usr/bin/env python3

import numpy as np
import scipy.sparse as sps
from scipy.special import expit
from time import time
from datetime import timedelta
from helper import TailBoost
from evaluation import evaluate_algorithm
from Base.Recommender_utils import similarityMatrixTopK


class SLIM_BPR_Recommender:
    """SLIM_BPR recommender with cosine similarity and no shrinkage"""

    def __init__(self):
        self.urm = None
        self.n_users = None
        self.n_items = None
        self.tb = None
        self.eligible_users = None
        self.learning_rate = None
        self.epochs = None
        self.similarity_matrix = None
        self.urm_validation = None

    def sample_triplet(self):
        # By randomly selecting a user in this way we could end up
        # with a user with no interactions
        # user_id = np.random.randint(0, n_users)
        user_id = np.random.choice(self.eligible_users)
        # Get user seen items and choose one
        user_seen_items = self.urm[user_id, :].indices
        pos_item_id = np.random.choice(user_seen_items)
        neg_item_selected = False
        # It's faster to just try again then to build a mapping of the non-seen items
        neg_item_id = None
        while not neg_item_selected:
            neg_item_id = np.random.randint(0, self.n_items)
            if neg_item_id not in user_seen_items:
                neg_item_selected = True
        return user_id, pos_item_id, neg_item_id

    def epoch_iteration(self):

        # Get number of available interactions
        num_positive_interactions = int(self.urm.nnz * 0.01)

        batch_size = 5000
        start_time = time()
        start_time_batch = time()

        # Uniform user sampling without replacement
        for num_sample in range(num_positive_interactions):
            # Sample
            user_id, positive_item_id, negative_item_id = self.sample_triplet()
            user_seen_items = self.urm[user_id, :].indices
            # Prediction
            x_i = self.similarity_matrix[positive_item_id, user_seen_items].sum()
            x_j = self.similarity_matrix[negative_item_id, user_seen_items].sum()
            # Gradient
            x_ij = x_i - x_j
            gradient = expit(-x_ij)
            #gradient = 1 / (1 + np.exp(x_ij))
            # Update
            self.similarity_matrix[positive_item_id, user_seen_items] += self.learning_rate * gradient
            self.similarity_matrix[positive_item_id, positive_item_id] = 0
            self.similarity_matrix[negative_item_id, user_seen_items] -= self.learning_rate * gradient
            self.similarity_matrix[negative_item_id, negative_item_id] = 0
            if num_sample % batch_size == 0 and num_sample > 0:
                elapsed = timedelta(seconds=int(time() - start_time))
                samples_ps = batch_size / (time() - start_time_batch)
                eta = timedelta(seconds=int((num_positive_interactions - num_sample) / samples_ps))
                print('Processed {0:7.0f} samples ( {1:5.2f}% ) in {2}. Samples/s: {3:4.0f}. ETA: {4}'.format(
                    num_sample,
                    100.0 * float(num_sample) / num_positive_interactions,
                    elapsed,
                    samples_ps,
                    eta))
                start_time_batch = time()

    def fit(self, urm, urm_validation=None, learning_rate=0.01, epochs=100):
        self.urm = urm.tocsr()
        if urm_validation is not None:
            self.urm_validation = urm_validation.tocsr()
        self.n_users = self.urm.shape[0]
        self.n_items = self.urm.shape[1]
        self.similarity_matrix = np.zeros((self.n_items, self.n_items))
        self.tb = TailBoost(self.urm)
        # Extract users having at least one interaction to choose from
        self.eligible_users = []
        for user_id in range(self.n_users):
            start_pos = self.urm.indptr[user_id]
            end_pos = self.urm.indptr[user_id + 1]
            if len(self.urm.indices[start_pos:end_pos]) > 0:
                self.eligible_users.append(user_id)
        self.learning_rate = learning_rate
        self.epochs = epochs

        start_time_train = time()

        last_MAP = 0
        for currentEpoch in range(self.epochs):
            start_time_epoch = time()
            self.epoch_iteration()
            if urm_validation is not None:
                self.similarity_matrix = sps.csr_matrix(self.similarity_matrix.T)
                self.similarity_matrix.eliminate_zeros()
                MAP = evaluate_algorithm(urm_validation, self)['MAP']
                self.similarity_matrix = self.similarity_matrix.todense().T
                if MAP < last_MAP:
                    print("Early stopping at {0}".format(currentEpoch))
                    break
                last_MAP = MAP
            elapsed_time = timedelta(seconds=int(time() - start_time_epoch))
            print("Epoch {0} of {1} complete in {2}.".format(currentEpoch + 1, epochs, elapsed_time))

        print("Train completed in {:.2f} minutes".format(int(time()-start_time_train)/60))

        self.similarity_matrix = self.similarity_matrix.T
        self.similarity_matrix = sps.csr_matrix(self.similarity_matrix)
        self.similarity_matrix = similarityMatrixTopK(self.similarity_matrix, k=100).tocsr()
        self.similarity_matrix.eliminate_zeros()

    def recommend(self, user_id, at=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.urm[user_id]
        scores = user_profile.dot(self.similarity_matrix)
        scores = scores.toarray().ravel()
        if exclude_seen:
            scores = self.filter_seen(user_id, scores)
        # rank items
        ranking = scores.argsort()[::-1]
        return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.urm.indptr[user_id]
        end_pos = self.urm.indptr[user_id + 1]
        user_profile = self.urm.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores
