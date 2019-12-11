#!/usr/bin/env python3

import numpy as np
from run_utils import build_all_matrices, train_test_split, evaluate, export, SplitType
from evaluation import multiple_evaluation
from list_merge import round_robin_list_merger, frequency_list_merger, medrank
from cf import ItemCFKNNRecommender, UserCFKNNRecommender
from cbf import ItemCBFKNNRecommender
#from slim_bpr import SLIM_BPR
from cython_modules.SLIM_BPR.SLIM_BPR_CYTHON import SLIM_BPR
from basic_recommenders import TopPopRecommender
from enum import Enum
import pprint as pp
import matplotlib.pyplot as plt


class MergingTechniques(Enum):
    WEIGHTS = 1
    RR = 2
    FREQ = 3
    MEDRANK = 4
    SLOTS = 5


class HybridRecommender:

    def __init__(self, recommenders, merging_type, weights=None):
        self.recommenders = recommenders
        self.weights = weights
        self.n_recommenders = len(recommenders)
        if merging_type == MergingTechniques.WEIGHTS:
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

    def get_scores(self, user_id, exclude_seen=True):
        assert self.weights is not None
        scores = [recommender.get_scores(user_id, exclude_seen) for recommender in self.recommenders]
        scores = [scores[i] * self.weights[i] for i in range(self.n_recommenders)]
        scores = np.array(scores)
        scores = scores.sum(axis=0)
        return scores

    def recommend_weights(self, user_id, at=10, exclude_seen=True):
        assert self.weights is not None
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


def ranged_rand(low=0.0, high=1.0):
    assert 1.0 >= high >= low >= 0.0
    x = np.random.rand()
    if low <= x <= high:
        return x
    return ranged_rand(low, high)


if __name__ == '__main__':

    EXPORT = False
    urm, icm, target_users = build_all_matrices()
    if EXPORT:
        urm_train = urm.tocsr()
        urm_test = None
    else:
        urm_train, urm_test = train_test_split(urm, SplitType.LOO_CYTHON)
    n_users, n_items = urm_train.shape

    tp_rec = TopPopRecommender()
    tp_rec.fit(urm_train)

    cf = ItemCFKNNRecommender(fallback_recommender=tp_rec)
    cf.fit(urm_train, top_k=5, shrink=20, similarity='tanimoto')

    slim = SLIM_BPR(fallback_recommender=tp_rec)
    slim.fit(urm_train, epochs=120)

    user_cf = UserCFKNNRecommender(fallback_recommender=tp_rec)
    user_cf.fit(urm_train, top_k=715, shrink=60, normalize=True, similarity='tanimoto')

    cbf_rec = ItemCBFKNNRecommender()
    cbf_rec.fit(urm_train, icm)

    weights_list = []
    maps = []
    x = []
    for i in range(100):
        x.append(i)
        weights = [
            ranged_rand(0.5, 1.0),  # Item-CF
            ranged_rand(0.3, 1.0),  # User-CF
            ranged_rand(0.5, 1.0),  # SLIM
            ranged_rand(0.0, 1.0),  # CBF
        ]
        hybrid = HybridRecommender([cf, user_cf, slim, cbf_rec], merging_type=MergingTechniques.WEIGHTS, weights=weights)
        result = evaluate(hybrid, urm_test)['MAP']
        maps.append(result)
        plt.scatter(x, maps)
        plt.show()
        print(weights, result)
        weights_list.append((weights, result))
    print(sorted(weights_list, key=lambda x: x[1]))
