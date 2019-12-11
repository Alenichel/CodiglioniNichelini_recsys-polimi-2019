#!/usr/bin/env python3

import numpy as np
import scipy.sparse as sps
from scipy.special import expit
from tqdm import trange
from basic_recommenders import TopPopRecommender
from helper import TailBoost
from Base.Recommender_utils import similarityMatrixTopK
from run_utils import build_all_matrices, train_test_split, SplitType, evaluate, export
from cython_modules.SLIM_BPR.SLIM_BPR_CYTHON import SLIM_BPR as CYTHON_SLIM_BPR
import matplotlib.pyplot as plt


class SLIM_BPR:
    """
    This class is a python porting of the BPRSLIM algorithm in MyMediaLite written in C#
    The code is identical with no optimizations
    """

    def __init__(self,
                 lambda_i=0.0025,
                 lambda_j=0.00025,
                 learning_rate=0.05,
                 fallback_recommender=None,
                 use_tailboost=False):

        self.urm_train = None
        self.n_users = None
        self.n_items = None
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j
        self.learning_rate = learning_rate
        self.S = None
        self.W = None
        self.fallback_recommender = fallback_recommender
        self.use_tailboost = use_tailboost
        self.tb = None

    def __str__(self):
        return 'SLIM BPR'

    def fit(self, urm_train, epochs=30, top_k=100):
        self.urm_train = urm_train.tocsr()
        self.n_users = urm_train.shape[0]
        self.n_items = urm_train.shape[1]
        self.S = np.zeros((self.n_items, self.n_items), dtype=float)
        for _ in trange(epochs, desc='Epochs'):
            self.epoch_iteration()
        # The similarity matrix is learnt row-wise
        # To be used in the product URM*S must be transposed to be column-wise
        self.W = self.S.T
        del self.S
        # TODO: Check
        self.W = similarityMatrixTopK(self.W, k=top_k, verbose=True).tocsr()
        self.W = sps.csr_matrix(self.W)
        self.W.eliminate_zeros()
        self.tb = TailBoost(urm_train)

    def epoch_iteration(self):
        num_positive_interactions = int(self.urm_train.nnz * 0.01)
        for _ in trange(num_positive_interactions, desc='Epoch iteration'):
            user_id, pos_item_id, neg_item_id = self.sample_triple()
            self.update_factors(user_id, pos_item_id, neg_item_id)

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

    def get_scores(self, user_id, exclude_seen=True):
        user_profile = self.urm_train[user_id]
        scores = user_profile.dot(self.W).toarray().ravel()
        if exclude_seen:
            scores = self.filter_seen(user_id, scores)
        return scores

    def recommend(self, user_id, at=None, exclude_seen=True):
        user_profile = self.urm_train[user_id]
        if self.fallback_recommender and user_profile.nnz == 0:
            return self.fallback_recommender.recommend(user_id, at, exclude_seen)
        else:
            scores = self.get_scores(user_id, exclude_seen)
            if self.use_tailboost:
                scores = self.tb.update_scores(scores)
            ranking = scores.argsort()[::-1]
            return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.urm_train.indptr[user_id]
        end_pos = self.urm_train.indptr[user_id + 1]
        user_profile = self.urm_train.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores


if __name__ == '__main__':
    '''
    EXPORT = False
    urm, icm, target_users = build_all_matrices()
    if EXPORT:
        urm_train = urm.tocsr()
        urm_test = None
    else:
        urm_train, urm_test = train_test_split(urm, SplitType.LOO_CYTHON)
    cbf_rec = ItemCBFKNNRecommender()
    cbf_rec.fit(urm_train, icm)
    tp_rec = TopPopRecommender()
    tp_rec.fit(urm_train)
    slim_rec = CYTHON_SLIM_BPR(use_tailboost=False, fallback_recommender=tp_rec)
    slim_rec.fit(urm_train, epochs=100)
    if EXPORT:
        export(target_users, slim_rec)
    else:
        evaluate(slim_rec, urm_test)
    '''
    urm, _, _ = build_all_matrices()
    urm_train, urm_test = train_test_split(urm, SplitType.PROBABILISTIC)
    top_pop = TopPopRecommender()
    top_pop.fit(urm_train)
    top_k_ticks = []
    maps_at_top_k = []
    while True:
        top_k = np.random.randint(1000)
        top_k_ticks.append(top_k)
        print('Evaluating top_k={0}'.format(top_k))
        slim = CYTHON_SLIM_BPR(fallback_recommender=top_pop)
        slim.fit(urm_train, epochs=1, top_k=top_k)
        maps_at_top_k.append(evaluate(slim, urm_test, cython=True)['MAP'])
        results = [(top_k_ticks[k], maps_at_top_k[k]) for k in range(len(top_k_ticks))]
        results = sorted(results, key=lambda x: x[1])[::-1]
        print(results)
        plt.scatter(top_k_ticks, maps_at_top_k)
        plt.show()
