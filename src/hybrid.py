#!/usr/bin/env python3

import numpy as np
from run_utils import build_all_matrices, train_test_split, evaluate, export, SplitType
from list_merge import round_robin_list_merger, frequency_list_merger, medrank
from cf import ItemCFKNNRecommender, UserCFKNNRecommender
from cbf import ItemCBFKNNRecommender
from cython_modules.SLIM_BPR.SLIM_BPR_CYTHON import SLIM_BPR
from basic_recommenders import TopPopRecommender
from enum import Enum


class MergingTechniques(Enum):
    WEIGHTS = 1
    RR = 2
    FREQ = 3
    MEDRANK = 4
    SLOTS = 5


class HybridRecommender:

    def __init__(self, recommenders, urm_train, merging_type, weights=None, fallback_recommender=None):
        self.recommenders = recommenders
        self.weights = weights
        self.n_recommenders = len(recommenders)
        self.urm_train = urm_train
        self.fallback_recommender = fallback_recommender
        if merging_type == MergingTechniques.WEIGHTS:
            assert self.weights is not None and len(self.weights) == self.n_recommenders
            self.recommend = self.recommend_weights
        elif merging_type == MergingTechniques.RR:
            self.recommend = self.recommend_lists_rr
        elif merging_type == MergingTechniques.FREQ:
            self.recommend = self.recommend_lists_freq
        elif merging_type == MergingTechniques.MEDRANK:
            self.recommend = self.recommend_lists_medrank
        elif merging_type == MergingTechniques.SLOTS:
            self.recommend = self.reccomend_excluding_from_cf
        else:
            raise ValueError('merging_type is not an instance of MergingTechnique')

    def __str__(self):
        return 'Hybrid'

    def get_scores(self, user_id, exclude_seen=True):
        scores = [recommender.get_scores(user_id, exclude_seen) for recommender in self.recommenders]
        scores = [scores[i] * self.weights[i] for i in range(self.n_recommenders)]
        scores = np.array(scores)
        scores = scores.sum(axis=0)
        return scores

    def recommend_weights(self, user_id, at=10, exclude_seen=True):
        user_profile = self.urm_train[user_id]
        if user_profile.nnz == 0 and self.fallback_recommender:
            return self.fallback_recommender.recommend(user_id, at=at, exclude_seen=exclude_seen)
        scores = self.get_scores(user_id, exclude_seen)
        ranking = scores.argsort()[::-1]
        return ranking[:at]

    def recommend_lists_rr(self, user_id, at=10, exclude_seen=True):
        recommendations = [recommender.recommend(user_id, at=at, exclude_seen=exclude_seen) for recommender in self.recommenders]
        return round_robin_list_merger(recommendations)[:at]

    def recommend_lists_freq(self, user_id, at=10, exclude_seen=True):
        recommendations = [recommender.recommend(user_id, at=at, exclude_seen=exclude_seen) for recommender in self.recommenders]
        return frequency_list_merger(recommendations)[:at]

    def recommend_lists_medrank(self, user_id, at=10, exclude_seen=True):
        recommendations = [recommender.recommend(user_id, at=at, exclude_seen=exclude_seen) for recommender in self.recommenders]
        return medrank(recommendations)[:at]

    def reccomend_excluding_from_cf(self, user_id, at=10, exclude_seen=True, slot_for_cf=6):
        reccomdations1 = self.recommenders[0].recommend(user_id, at=at, exclude_seen=exclude_seen)
        reccomdations2 = self.recommenders[1].recommend(user_id, at=at, exclude_seen=exclude_seen)
        f = reccomdations1[:slot_for_cf]
        s = reccomdations2
        l = []
        for e in f:
            l = l + [e]
        for e in s:
            l = l + [e]
        l = list(dict.fromkeys(l))
        return l[:at]


if __name__ == '__main__':
    EXPORT = False
    urm, icm, ucm, target_users = build_all_matrices()
    if EXPORT:
        urm_train = urm.tocsr()
        urm_test = None
    else:
        urm_train, urm_test = train_test_split(urm, SplitType.PROBABILISTIC)
    n_users, n_items = urm_train.shape

    tp_rec = TopPopRecommender()
    tp_rec.fit(urm_train)

    item_cf = ItemCFKNNRecommender(fallback_recommender=tp_rec)
    item_cf.fit(urm_train, top_k=5, shrink=20, similarity='tanimoto')

    slim = SLIM_BPR(fallback_recommender=tp_rec)
    slim.fit(urm_train, epochs=300)

    user_cf = UserCFKNNRecommender(fallback_recommender=tp_rec)
    user_cf.fit(urm_train, top_k=715, shrink=60, normalize=True, similarity='tanimoto')

    cbf = ItemCBFKNNRecommender()
    cbf.fit(urm_train, icm)

    hybrid = HybridRecommender([cbf, item_cf, slim, user_cf], merging_type=MergingTechniques.WEIGHTS, weights=[3.048, 4.977, 4.956, 0.025])

    if EXPORT:
        export(target_users, hybrid)
    else:
        evaluate(hybrid, urm_test, cython=True)
