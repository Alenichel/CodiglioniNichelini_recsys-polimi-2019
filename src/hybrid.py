#!/usr/bin/env python3

import numpy as np
from run_utils import build_all_matrices, train_test_split, evaluate, export, SplitType
from list_merge import round_robin_list_merger, frequency_list_merger, medrank
from basic_recommenders import TopPopRecommender
from cf import ItemCFKNNRecommender
from cbf import ItemCBFKNNRecommender
from slim_bpr import SLIM_BPR


class HybridRecommender:

    def __init__(self, recommenders, weights=None):
        self.recommenders = recommenders
        self.weights = weights
        self.n_recommenders = len(recommenders)
        self.recommend = self.recommend_lists_medrank
        print(self.recommend)

    def recommend_weights(self, user_id, at=10, exclude_seen=True):
        assert self.weights is not None
        scores = [recommender.get_scores(user_id, exclude_seen) for recommender in self.recommenders]
        scores = [scores[i] * self.weights[i] for i in range(self.n_recommenders)]
        scores = np.array(scores)
        scores = scores.sum(axis=0)
        ranking = scores.argsort()[::-1]
        return ranking[:at]

    def recommend_lists_rr(self, user_id, at=10, exclude_seen=True):
        recommendations = [recommender.recommend(user_id, exclude_seen) for recommender in self.recommenders]
        return round_robin_list_merger(recommendations)[:at]

    def recommend_lists_freq(self, user_id, at=10, exclude_seen=True):
        recommendations = [recommender.recommend(user_id, exclude_seen) for recommender in self.recommenders]
        return frequency_list_merger(recommendations)[:at]

    def recommend_lists_medrank(self, user_id, at=10, exclude_seen=True):
        recommendations = [recommender.recommend(user_id, exclude_seen) for recommender in self.recommenders]
        return medrank(recommendations)[:at]


if __name__ == '__main__':
    EXPORT = False
    urm, icm, target_users = build_all_matrices()
    if EXPORT:
        urm_train = urm.tocsr()
        urm_test = None
    else:
        urm_train, urm_test = train_test_split(urm, SplitType.LOO)
    n_users, n_items = urm_train.shape
    tp_rec = TopPopRecommender()
    tp_rec.fit(urm_train)
    cbf_rec = ItemCBFKNNRecommender()
    cbf_rec.fit(urm_train, icm, top_k=5, shrink=20, similarity='tanimoto')
    cf_rec = ItemCFKNNRecommender()
    cf_rec.fit(urm_train, top_k=5, shrink=20, similarity='tanimoto')
    slim_rec = SLIM_BPR(use_tailboost=True)
    slim_rec.fit(urm_train, epochs=100)
    rec = HybridRecommender([cf_rec, slim_rec, cbf_rec, tp_rec])
    if EXPORT:
        export(target_users, rec)
    else:
        evaluate(rec, urm_test)
