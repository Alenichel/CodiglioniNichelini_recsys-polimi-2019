#!/usr/bin/env python3

import numpy as np
from run_utils import build_all_matrices, train_test_split, evaluate, export, SplitType
from list_merge import round_robin_list_merger, frequency_list_merger, medrank
from cf import ItemCFKNNRecommender
from cbf import ItemCBFKNNRecommender
from slim_bpr import SLIM_BPR
from basic_recommenders import TopPopRecommender
from enum import Enum


class MergingTechniques(Enum):
    WEIGHTS = 1
    RR = 2
    FREQ = 3


class HybridRecommender:

    def __init__(self, recommenders, merging_type, weights=None):
        self.recommenders = recommenders
        self.weights = weights
        self.n_recommenders = len(recommenders)
        if merging_type == MergingTechniques.WEIGHTS:
            self.recommend = self.recommend_weights
        elif merging_type == MergingTechniques.RR:
            self.recommend = self.recommend_lists_rr
        else:
            self.recommend = self.recommend_lists_freq
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
        recommendations = [recommender.recommend(user_id, exclude_seen=exclude_seen) for recommender in self.recommenders]
        return round_robin_list_merger(recommendations, at=at)[:at]

    def recommend_lists_freq(self, user_id, at=10, exclude_seen=True):
        recommendations = [recommender.recommend(user_id, exclude_seen=exclude_seen) for recommender in self.recommenders]
        return frequency_list_merger(recommendations)[:at]

    def recommend_lists_medrank(self, user_id, at=10, exclude_seen=True):
        recommendations = [recommender.recommend(user_id, exclude_seen=exclude_seen) for recommender in self.recommenders]
        return medrank(recommendations)[:at]


if __name__ == '__main__':
    TUNER = False
    if TUNER:
        ROUND = 5
        urm, icm, target_users = build_all_matrices()
        cumulativeMAP = 0
        results = []
        for x in range(ROUND):
            urm_train, urm_test = train_test_split(urm, SplitType.LOO)
            n_users, n_items = urm_train.shape
            top_pop = TopPopRecommender()
            top_pop.fit(urm_train)
            cf_rec1 = ItemCFKNNRecommender(fallback_recommender=top_pop)
            cf_rec1.fit(urm_train, top_k=5, shrink=20, similarity='tanimoto')
            cf_rec2 = ItemCFKNNRecommender(fallback_recommender=top_pop)
            cf_rec2.fit(urm_train, top_k=5, shrink=35, similarity='cosine')
            cf_rec3 = ItemCFKNNRecommender(fallback_recommender=top_pop)
            cf_rec3.fit(urm_train, top_k=10, shrink=20, similarity='asymmetric')
            rec = HybridRecommender([cf_rec1, cf_rec2, cf_rec3], merging_type=MergingTechniques.WEIGHTS,
                                    weights=[1 / 3, 1 / 3, 1 / 3])
            roundMAP = evaluate(rec, urm_test)['MAP']
            results.append(roundMAP)
            cumulativeMAP += roundMAP
            print("median value so far %f (ROUND %d)" % (cumulativeMAP / (x + 1), x + 1))
        print(results.sort())
    else:
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
        slim_rec.fit(urm_train, epochs=1)
        rec = HybridRecommender([cf_rec, slim_rec, cbf_rec, tp_rec], merging_type=MergingTechniques.RR)
        if EXPORT:
            export(target_users, rec)
        else:
            evaluate(rec, urm_test)
