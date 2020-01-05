#!/usr/bin/env python3

import numpy as np
from run_utils import set_seed, build_all_matrices, build_age_ucm, train_test_split, evaluate, export, SplitType, evaluate_mp
from list_merge import round_robin_list_merger, frequency_list_merger, medrank
from cf import ItemCFKNNRecommender, UserCFKNNRecommender
from cbf import ItemCBFKNNRecommender, UserCBFKNNRecommender
from cython_modules.SLIM_BPR.SLIM_BPR_CYTHON import SLIM_BPR
from basic_recommenders import TopPopRecommender
from enum import Enum
from slim_elasticnet import SLIMElasticNetRecommender
from mf import AlternatingLeastSquare
from model_hybrid import ModelHybridRecommender
from bayes_opt import BayesianOptimization
from similaripy_rs import SimPyRecommender


class MergingTechniques(Enum):
    WEIGHTS = 1
    RR = 2
    FREQ = 3
    MEDRANK = 4


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
        else:
            raise ValueError('merging_type is not an instance of MergingTechnique')
        n_users, n_items = urm_train.shape
        self.popular_items = np.flip(np.argsort(np.array(urm_train.sum(axis=0)).squeeze()), axis=0)[: int(n_items * 0.8)]

    def __str__(self):
        return 'Hybrid'

    def get_scores(self, user_id, exclude_seen=True):
        scores = [recommender.get_scores(user_id, exclude_seen) for recommender in self.recommenders]
        scores = [scores[i] * self.weights[i] for i in range(self.n_recommenders)]
        scores = np.array(scores)
        scores = scores.sum(axis=0)
        return scores

    def recommend_weights(self, user_id, at=10, exclude_seen=True, remove_least_popular=False):
        user_profile = self.urm_train[user_id]
        if user_profile.nnz == 0 and self.fallback_recommender:
            return self.fallback_recommender.recommend(user_id, at=at, exclude_seen=exclude_seen)
        scores = self.get_scores(user_id, exclude_seen)
        ranking = scores.argsort()[::-1]
        if remove_least_popular:
            is_relevant = np.in1d(ranking, self.popular_items, assume_unique=True)
            ranking = ranking[is_relevant]
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


def get_hybrid_components(urm_train, icm, ucm, cache=True):
    # TOP-POP
    top_pop = TopPopRecommender()
    top_pop.fit(urm_train)
    # USER CBF
    user_cbf = UserCBFKNNRecommender()
    user_cbf.fit(urm_train, ucm, top_k=496, shrink=0, normalize=False)
    # HYBRID FALLBACK
    hybrid_fb = HybridRecommender([top_pop, user_cbf], urm_train, merging_type=MergingTechniques.MEDRANK)
    # ITEM CF
    item_cf = ItemCFKNNRecommender(fallback_recommender=hybrid_fb)
    item_cf.fit(urm_train, top_k=4, shrink=34, normalize=False, similarity='jaccard')
    # SLIM BPR
    slim_bpr = SLIM_BPR(fallback_recommender=hybrid_fb)
    slim_bpr.fit(urm_train, epochs=300)
    # SLIM ELASTICNET
    slim_enet = SLIMElasticNetRecommender(fallback_recommender=hybrid_fb)
    slim_enet.fit(urm_train, cache=cache)
    # MODEL HYBRID
    model_hybrid = ModelHybridRecommender([item_cf.w_sparse, slim_bpr.W, slim_enet.W_sparse], [42.82, 535.4, 52.17],
                                          fallback_recommender=hybrid_fb)
    model_hybrid.fit(urm_train, top_k=977)
    # USER CF
    user_cf = UserCFKNNRecommender()
    user_cf.fit(urm_train, top_k=593, shrink=4, normalize=False, similarity='tanimoto')
    # ITEM CBF
    item_cbf = ItemCBFKNNRecommender()
    item_cbf.fit(urm_train, icm, 417, 0.3, normalize=True)
    # ALS
    als = AlternatingLeastSquare()
    als.fit(urm_train, n_factors=868, regularization=99.75, iterations=152, cache=cache)
    # RP3BETA
    rp3beta = SimPyRecommender()
    rp3beta.fit(urm_train)
    return hybrid_fb, model_hybrid, user_cf, item_cbf, als, rp3beta


def get_hybrid(urm_train, icm, ucm, cache=True):
    hybrid_fb, model_hybrid, user_cf, item_cbf, als, rp3beta = get_hybrid_components(urm_train, icm, ucm, cache)
    hybrid = HybridRecommender([model_hybrid, user_cf, item_cbf, als, rp3beta],
                               urm_train,
                               merging_type=MergingTechniques.WEIGHTS,
                               weights=[0.4767, 2.199, 2.604, 7.085, 0.04029],
                               fallback_recommender=hybrid_fb)
    return hybrid


def to_optimize(w_mh, w_ucf, w_icbf, w_als, w_rp3):
    hybrid = HybridRecommender([model_hybrid, user_cf, item_cbf, als, rp3beta],
                               urm_train,
                               merging_type=MergingTechniques.WEIGHTS,
                               weights=[w_mh, w_ucf, w_icbf, w_als, w_rp3],
                               fallback_recommender=hybrid_fb)
    return evaluate(hybrid, urm_test, verbose=False)['MAP']


def to_optimize_fixed(w_rp3):
    hybrid = HybridRecommender([model_hybrid, user_cf, item_cbf, als, rp3beta],
                               urm_train,
                               merging_type=MergingTechniques.WEIGHTS,
                               weights=[0.4767, 2.199, 2.604, 7.085, w_rp3],
                               fallback_recommender=hybrid_fb)
    return evaluate(hybrid, urm_test, verbose=False)['MAP']


if __name__ == '__main__':
    set_seed(42)
    EXPORT = False
    urm, icm, ucm, target_users = build_all_matrices()
    age_ucm = build_age_ucm(urm.shape[0])
    if EXPORT:
        urm_train = urm.tocsr()
        urm_test = None
    else:
        urm_train, urm_test = train_test_split(urm)
        '''ten_percent_test = [9/10, 8/9, 7/8]
        fifteen_percent_test = [17/20, 14/17, 11/14]
        # The splitting are done so that each test set is 15% if the original data set
        # and the final train set is 55% of the original data set
        urm_train, urm_test1 = train_test_split(urm, SplitType.PROBABILISTIC, split=fifteen_percent_test[0])
        urm_train, urm_test2 = train_test_split(urm_train, SplitType.PROBABILISTIC, split=fifteen_percent_test[1])
        urm_train, urm_test3 = train_test_split(urm_train, SplitType.PROBABILISTIC, split=fifteen_percent_test[1])'''

    hybrid = get_hybrid(urm_train, icm, ucm, cache=not EXPORT)

    if EXPORT:
        export(target_users, hybrid)
    else:
        result = evaluate(hybrid, urm_test)
        print(result)
    exit()

    hybrid_fb, model_hybrid, user_cf, item_cbf, als, rp3beta = get_hybrid_components(urm_train, icm, ucm)

    pbounds = {
        #'w_mh': (0.5, 1),
        #'w_ucf': (2, 2.5),
        #'w_icbf': (2.7, 3.2),
        #'w_als': (6.5, 8),
        'w_rp3': (0, 10)
    }

    optimizer = BayesianOptimization(
        f=to_optimize_fixed,
        pbounds=pbounds,
    )

    optimizer.maximize(
        init_points=50,
        n_iter=250,
    )

    print(optimizer.max)
