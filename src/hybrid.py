#!/usr/bin/env python3

import numpy as np
from run_utils import build_all_matrices, train_test_split, evaluate, export, SplitType
from evaluation import multiple_evaluation
from list_merge import round_robin_list_merger, frequency_list_merger, medrank
from cf import ItemCFKNNRecommender, UserCFKNNRecommender
from cbf import ItemCBFKNNRecommender
from slim_bpr import SLIM_BPR
from basic_recommenders import TopPopRecommender
from enum import Enum
import pprint as pp


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
        recommendations = [recommender.recommend(user_id, at=at, exclude_seen=exclude_seen) for recommender in self.recommenders]
        return round_robin_list_merger(recommendations)[:at]

    def recommend_lists_freq(self, user_id, at=10, exclude_seen=True):
        recommendations = [recommender.recommend(user_id, at=at, exclude_seen=exclude_seen) for recommender in self.recommenders]
        return frequency_list_merger(recommendations)[:at]

    def recommend_lists_medrank(self, user_id, at=10, exclude_seen=True):
        recommendations = [recommender.recommend(user_id, at=at, exclude_seen=exclude_seen) for recommender in self.recommenders]
        return medrank(recommendations)[:at]

    def reccomend_excluding_from_cf(self, user_id, at=10, exclude_seen=True, slot_for_cf=8):
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
    TUNER = 7

    if TUNER == 0: # BEST ENTRY SO FAR
        EXPORT = True
        urm, icm, target_users = build_all_matrices()
        if EXPORT:
            urm_train = urm.tocsr()
            urm_test = None
        else:
            urm_train, urm_test = train_test_split(urm, SplitType.LOO)
        n_users, n_items = urm_train.shape
        tp_rec = TopPopRecommender()
        tp_rec.fit(urm_train)
        cf_rec1 = ItemCFKNNRecommender(fallback_recommender=tp_rec)
        cf_rec1.fit(urm_train, top_k=5, shrink=20, similarity='tanimoto')
        cf_rec2 = ItemCFKNNRecommender(fallback_recommender=tp_rec)
        cf_rec2.fit(urm_train, top_k=5, shrink=35, similarity='cosine')
        cf_rec3 = ItemCFKNNRecommender(fallback_recommender=tp_rec)
        cf_rec3.fit(urm_train, top_k=10, shrink=20, similarity='asymmetric')
        rec = HybridRecommender([cf_rec1, cf_rec2, cf_rec3], merging_type=MergingTechniques.RR)
        if EXPORT:
            export(target_users, rec)
        else:
            evaluate(rec, urm_test)

    elif TUNER == 1:
        ROUND = 5
        urm, icm, target_users = build_all_matrices()
        cumulativeMAP = 0
        results = []
        for x in range(ROUND):
            urm_train, urm_test = train_test_split(urm, SplitType.LOO)
            n_users, n_items = urm_train.shape
            top_pop = TopPopRecommender()
            top_pop.fit(urm_train)
            cf_rec1 = ItemCFKNNRecommender(fallback_recommender=top_pop, use_tail_boost=True)
            cf_rec1.fit(urm_train, top_k=5, shrink=20, similarity='tanimoto')
            cf_rec2 = ItemCFKNNRecommender(fallback_recommender=top_pop, use_tail_boost=True)
            cf_rec2.fit(urm_train, top_k=5, shrink=35, similarity='cosine')
            cf_rec3 = ItemCFKNNRecommender(fallback_recommender=top_pop, use_tail_boost=True)
            cf_rec3.fit(urm_train, top_k=10, shrink=20, similarity='asymmetric')
            rec = HybridRecommender([cf_rec1, cf_rec2, cf_rec3], merging_type=MergingTechniques.RR)
            roundMAP = evaluate(rec, urm_test)['MAP']
            results.append(roundMAP)
            cumulativeMAP += roundMAP
            print("median value so far %f (ROUND %d)" % (cumulativeMAP / (x + 1), x + 1))
        print(results.sort())

    elif TUNER == 2:
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
        slim_rec = SLIM_BPR()
        slim_rec.fit(urm_train, epochs=10)
        rec = HybridRecommender([cf_rec, slim_rec, cbf_rec, tp_rec], merging_type=MergingTechniques.RR)
        if EXPORT:
            export(target_users, rec)
        else:
            evaluate(rec, urm_test)

    elif TUNER == 3:  #Hybrid fallback recommendation system (very bad idea)
        EXPORT = True
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
        h = HybridRecommender([cbf_rec, tp_rec], merging_type=MergingTechniques.RR)

        cf_rec = ItemCFKNNRecommender(fallback_recommender=h)
        cf_rec.fit(urm_train, top_k=5, shrink=20, similarity='tanimoto')
        if EXPORT:
            export(target_users, cf_rec)
        else:
            evaluate(cf_rec, urm_test)

    # Tuner for hybridization
    elif TUNER == 4:
        ROUND = 5
        urm, icm, target_users = build_all_matrices()
        results = []
        for x in range(0, 100, 5):
            weights = [x/100, 1 - x/100]
            cumulativeMAP = 0
            median_value = 0
            for n in range(ROUND):
                urm_train, urm_test = train_test_split(urm, SplitType.LOO)
                n_users, n_items = urm_train.shape
                tp_rec = TopPopRecommender()
                tp_rec.fit(urm_train)
                cf_rec1 = ItemCFKNNRecommender(fallback_recommender=tp_rec)
                cf_rec1.fit(urm_train, top_k=5, shrink=20, similarity='tanimoto')
                cf_rec2 = ItemCFKNNRecommender(fallback_recommender=tp_rec)
                cf_rec2.fit(urm_train, top_k=5, shrink=35, similarity='cosine')
                cf_rec3 = ItemCFKNNRecommender(fallback_recommender=tp_rec)
                cf_rec3.fit(urm_train, top_k=10, shrink=20, similarity='asymmetric')
                rec = HybridRecommender([cf_rec1, cf_rec2, cf_rec3], merging_type=MergingTechniques.RR)
                cbf_rec = ItemCBFKNNRecommender()
                cbf_rec.fit(urm_train, icm, top_k=5, shrink=20, similarity='tanimoto')
                final = HybridRecommender([rec, cbf_rec], merging_type=MergingTechniques.WEIGHTS, weights=weights)
                roundMAP = evaluate(final, urm_test)['MAP']
                cumulativeMAP += roundMAP
                median_value = cumulativeMAP / (n+1)
                print("Median value so far %f" % median_value)
            results.append({
                "value": median_value,
                "weights": weights
            })
        pp.print(results)

    elif TUNER == 5:
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
            rec = HybridRecommender([cf_rec1, cf_rec2, cf_rec3], merging_type=MergingTechniques.RR)
            slim_rec = SLIM_BPR()
            slim_rec.fit(urm_train, epochs=100)
            final = HybridRecommender([rec, slim_rec], merging_type=MergingTechniques.SLOTS)
            roundMAP = evaluate(final, urm_test)['MAP']
            results.append(roundMAP)
            cumulativeMAP += roundMAP
            print("median value so far %f (ROUND %d)" % (cumulativeMAP / (x + 1), x + 1))
        print(results.sort())

    elif TUNER == 6:
        EXPORT = True
        urm, icm, target_users = build_all_matrices()
        if EXPORT:
            urm_train = urm.tocsr()
            urm_test = None
        else:
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
        rec = HybridRecommender([cf_rec1, cf_rec2, cf_rec3], merging_type=MergingTechniques.RR)
        slim_rec = SLIM_BPR()
        slim_rec.fit(urm_train, epochs=100)
        final = HybridRecommender([rec, slim_rec], merging_type=MergingTechniques.SLOTS)
        if EXPORT:
            export(target_users, final)
        else:
            evaluate(final, urm_test)

    elif TUNER == 7:
        EXPORT = False
        urm, icm, target_users = build_all_matrices()
        if EXPORT:
            urm_train = urm.tocsr()
            urm_test = None
        else:
            urm_train, urm_test = train_test_split(urm, SplitType.LOO_CYTHON)
        n_users, n_items = urm_train.shape
        top_pop = TopPopRecommender()
        top_pop.fit(urm_train)
        user_cf = UserCFKNNRecommender(fallback_recommender=top_pop)
        user_cf.fit(urm_train, top_k=645, shrink=539, normalize=True, similarity='cosine')
        item_cf = ItemCFKNNRecommender(fallback_recommender=top_pop)
        item_cf.fit(urm_train, top_k=5, shrink=20, normalize=True, similarity='tanimoto')
        hybrid = HybridRecommender([item_cf, user_cf], merging_type=MergingTechniques.RR)
        if EXPORT:
            export(target_users, hybrid)
        else:
            evaluate(hybrid, urm_test)
