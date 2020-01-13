#!/usr/bin/env python3

import numpy as np
from tqdm import trange

from run_utils import set_seed, build_all_matrices, train_test_split, evaluate, export, SplitType, multiple_splitting
from list_merge import round_robin_list_merger, frequency_list_merger, medrank
from cf import ItemCFKNNRecommender, UserCFKNNRecommender, get_item_cf, get_user_cf
from cbf import ItemCBFKNNRecommender, UserCBFKNNRecommender, get_item_cbf, get_user_cbf
from cython_modules.SLIM_BPR.SLIM_BPR_CYTHON import SLIM_BPR
from basic_recommenders import TopPopRecommender
from enum import Enum
from slim_elasticnet import SLIMElasticNetRecommender
from mf import AlternatingLeastSquare, get_als
from model_hybrid import ModelHybridRecommender, get_model_hybrid
from bayes_opt import BayesianOptimization
#from similaripy_rs import SimPyRecommender


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


def get_fallback(urm_train, ucm, generalized=False):
    top_pop = TopPopRecommender()
    top_pop.fit(urm_train)
    user_cbf = get_user_cbf(urm_train, ucm, generalized=generalized)
    hybrid_fb = HybridRecommender([user_cbf, top_pop], urm_train, merging_type=MergingTechniques.RR)
    return hybrid_fb


def get_hybrid_components(urm_train, icm, ucm, cache=True, fallback=True, generalized=False):
    fb = get_fallback(urm_train, ucm, generalized=generalized) if fallback else None
    model_hybrid = get_model_hybrid(urm_train, generalized=generalized)
    user_cf = get_user_cf(urm_train, generalized=generalized)
    item_cbf = get_item_cbf(urm_train, icm, generalized=generalized)
    als = get_als(urm_train, generalized=generalized, cache=cache)
    return fb, model_hybrid, user_cf, item_cbf, als


def get_hybrid(urm_train, icm, ucm, cache=True, fallback=True, generalized=False):
    fb, model_hybrid, user_cf, item_cbf, als = get_hybrid_components(urm_train, icm, ucm, cache, fallback, generalized=generalized)

    if generalized:
        hybrid = HybridRecommender([model_hybrid, user_cf, item_cbf, als],
                                   urm_train,
                                   merging_type=MergingTechniques.WEIGHTS,
                                   weights=[0.7242, 1.629, 0.9316, 8.0133],
                                   fallback_recommender=fb)

    else:
        hybrid = HybridRecommender([model_hybrid, user_cf, item_cbf, als],
                               urm_train,
                               merging_type=MergingTechniques.WEIGHTS,
                               weights=[0.4767, 2.199, 2.604, 7.085],
                               fallback_recommender=fb)
    return hybrid


def hybrid_multiple_evaluation():
    _, icm, ucm, _ = build_all_matrices()
    trains, tests, seeds = multiple_splitting()
    cumulative_MAP = 0
    for n in trange(len(trains)):
        hybrid = get_hybrid(trains[n], icm=icm, ucm=ucm, cache=True, generalized=True)
        cumulative_MAP = evaluate(hybrid, tests[n], cython=True)
    averageMAP = cumulative_MAP / len(trains)
    print('Average MAP: ', str(averageMAP))

def check_best(bests, icm, ucm):
    assert type(bests) == list
    trains, tests, seeds = multiple_splitting()

    fbs = list()
    model_hybrids = list()
    ucfs = list()
    icbfs = list()
    alss = list()

    for n in range(len(seeds)):
        fb = get_fallback(trains[n], ucm)
        fbs.append(fb)

        set_seed(seeds[n])
        model_hybrid = get_model_hybrid(trains[n], generalized=True)
        model_hybrids.append(model_hybrid)

        ucf = get_user_cf(trains[n], fb=fb, generalized=True)
        ucfs.append(ucf)
        icbf = get_item_cbf(trains[n], icm, generalized=True)
        icbfs.append(icbf)
        als = get_als(trains[n], fb=fb, generalized=True, cache=True)
        alss.append(als)

    set_seed(42)
    for best in bests:
        w_mh = best['params']['w_mh']
        w_ucf = best['params']['w_ucf']
        w_icbf = best['params']['w_icbf']
        w_als = best['params']['w_als']
        cumulative_MAP = 0
        for n in trange(len(trains)):
            hybrid = HybridRecommender([model_hybrids[n], ucfs[n], icbfs[n], alss[n]],
                                       trains[n],
                                       merging_type=MergingTechniques.WEIGHTS,
                                       weights=[w_mh, w_ucf, w_icbf, w_als],
                                       fallback_recommender=fbs[n])
            cumulative_MAP += evaluate(hybrid, tests[n], verbose=False)['MAP']
        averageMAP = cumulative_MAP / len(trains)
        best['AVG_MAP'] = averageMAP

    bests.sort(key=lambda dic: dic['AVG_MAP'], reverse=True)
    for best in bests:
        print(best)

    return bests


def tuner():
    urm, icm, ucm, _ = build_all_matrices()
    urm_train, urm_test = train_test_split(urm, SplitType.PROBABILISTIC)
    pbounds = {
        'w_mh': (0, 10),
        'w_ucf': (0, 10),
        'w_icbf': (0, 10),
        'w_als': (0, 10),
    }

    fb = get_fallback(urm_train, ucm)
    model_hybrid = get_model_hybrid(urm_train, generalized=True)
    user_cf = get_user_cf(urm_train, generalized=True)
    item_cbf = get_item_cbf(urm_train, icm, generalized=True)
    als = get_als(urm_train, generalized=False, cache=True)

    def to_optimize(w_mh, w_ucf, w_icbf, w_als):
        hybrid = HybridRecommender([model_hybrid, user_cf, item_cbf, als],
                                   urm_train,
                                   merging_type=MergingTechniques.WEIGHTS,
                                   weights=[w_mh, w_ucf, w_icbf, w_als],
                                   fallback_recommender=fb)
        return evaluate(hybrid, urm_test, verbose=False)['MAP']

    optimizer = BayesianOptimization(f=to_optimize, pbounds=pbounds)
    optimizer.probe(
        params={'w_mh': 0.4767, 'w_ucf': 2.199, 'w_icbf': 2.604, 'w_als': 7.085},
        lazy=True
    )
    optimizer.maximize(init_points=100, n_iter=300)
    opt_results = optimizer.res
    opt_results.sort(key=lambda dic: dic['target'], reverse=True)
    check_best(opt_results[:10], icm, ucm)


if __name__ == '__main__':
    set_seed(42)
    #tuner()
    #exit()
    EXPORT = False
    urm, icm, ucm, target_users = build_all_matrices()
    if EXPORT:
        urm_train = urm.tocsr()
        urm_test = None
    else:
        urm_train, urm_test = train_test_split(urm)

    hybrid = get_hybrid(urm_train, icm, ucm, cache=not EXPORT, generalized=True)

    if EXPORT:
        export(target_users, hybrid)
    else:
        evaluate(hybrid, urm_test)