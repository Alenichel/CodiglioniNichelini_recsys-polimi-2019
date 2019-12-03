#!/usr/bin/env python3

import numpy as np
import scipy.sparse as sps
from scipy.special import expit
from tqdm import trange
from run_utils import build_all_matrices, train_test_split, SplitType, evaluate, export
from helper import TailBoost
from Base.Recommender_utils import similarityMatrixTopK


class SLIM_BPR_Recommender:
    """SLIM_BPR recommender with cosine similarity and no shrinkage"""

    def __init__(self, use_tailboost=False, fallback_recommender=None):
        self.urm = None
        self.n_users = None
        self.n_items = None
        self.use_tailboost = use_tailboost
        self.tb = None
        self.eligible_users = None
        self.learning_rate = None
        self.epochs = None
        self.similarity_matrix = None
        self.fallback_recommender = fallback_recommender

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

        # Uniform user sampling without replacement
        for num_sample in trange(num_positive_interactions, desc='Epoch iteration'):
            # Sample
            user_id, positive_item_id, negative_item_id = self.sample_triplet()
            user_seen_items = self.urm[user_id, :].indices
            # Prediction
            x_i = self.similarity_matrix[positive_item_id, user_seen_items].sum()
            x_j = self.similarity_matrix[negative_item_id, user_seen_items].sum()
            # Gradient
            x_ij = x_i - x_j
            gradient = expit(-x_ij)
            # Update
            self.similarity_matrix[positive_item_id, user_seen_items] += self.learning_rate * gradient
            self.similarity_matrix[positive_item_id, positive_item_id] = 0
            self.similarity_matrix[negative_item_id, user_seen_items] -= self.learning_rate * gradient
            self.similarity_matrix[negative_item_id, negative_item_id] = 0

    def fit(self, urm, learning_rate=0.01, epochs=10):
        self.urm = urm.tocsr()
        self.n_users = self.urm.shape[0]
        self.n_items = self.urm.shape[1]
        self.similarity_matrix = np.zeros((self.n_items, self.n_items))
        self.tb = TailBoost(self.urm)
        # Extract users having at least one interaction to choose from
        self.eligible_users = []
        for user_id in trange(self.n_users, desc='Eligible users'):
            start_pos = self.urm.indptr[user_id]
            end_pos = self.urm.indptr[user_id + 1]
            if len(self.urm.indices[start_pos:end_pos]) > 0:
                self.eligible_users.append(user_id)
        self.learning_rate = learning_rate
        self.epochs = epochs

        for _ in trange(self.epochs, desc='Epochs'):
            self.epoch_iteration()

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
        if self.use_tailboost:
            scores = self.tb.update_scores(scores)
        # rank items
        ranking = scores.argsort()[::-1]
        return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.urm.indptr[user_id]
        end_pos = self.urm.indptr[user_id + 1]
        user_profile = self.urm.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores


if __name__ == '__main__':
    EXPORT = False
    urm, icm, target_users = build_all_matrices()
    if EXPORT:
        urm_train = urm
        urm_test = None
    else:
        urm_train, urm_test = train_test_split(urm, SplitType.LOO)
    #cbf_rec = ItemCBFKNNRecommender()
    #cbf_rec.fit(urm_train, icm)
    slim_rec = SLIM_BPR_Recommender(use_tailboost=True)
    slim_rec.fit(urm_train, epochs=100)
    if EXPORT:
        export(target_users, slim_rec)
    else:
        evaluate(slim_rec, urm_test)
