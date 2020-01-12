#!/usr/bin/env python3

import numpy as np
from tqdm import trange
from Base.Recommender_utils import similarityMatrixTopK
from run_utils import set_seed, build_all_matrices, train_test_split, SplitType, export, evaluate, multiple_splitting
from basic_recommenders import TopPopRecommender
from cf import get_item_cf
from cbf import get_user_cbf
from cython_modules.SLIM_BPR.SLIM_BPR_CYTHON import SLIM_BPR
from slim_elasticnet import SLIMElasticNetRecommender
from bayes_opt import BayesianOptimization


def get_model_hybrid(urm_train, generalized=False, cache=True):
    item_cf = get_item_cf(urm_train, generalized=generalized)
    slim_bpr = SLIM_BPR()
    slim_bpr.fit(urm_train, epochs=300)
    slim_enet = SLIMElasticNetRecommender()
    slim_enet.fit(urm_train, cache=cache)
    if generalized:
        model_hybrid = ModelHybridRecommender([item_cf.w_sparse, slim_bpr.W, slim_enet.W_sparse],
                                              [28.92, 373.11, 38.67])
        model_hybrid.fit(urm_train, top_k=541)
    else:
        model_hybrid = ModelHybridRecommender([item_cf.w_sparse, slim_bpr.W, slim_enet.W_sparse],
                                              [42.82, 535.4, 52.17])
        model_hybrid.fit(urm_train, top_k=977)
    return model_hybrid


class ModelHybridRecommender:

    def __init__(self, models, weights, fallback_recommender=None):
        assert len(models) >= 1 and len(models) == len(weights)
        for m in models:
            assert m.shape == models[0].shape
        self.models = np.array(models)
        self.weights = np.array(weights)
        self.urm = None
        self.w_sparse = None
        self.fallback_recommender = fallback_recommender
        self.use_tail_boost = False
        self.tb = None

    def fit(self, urm, top_k=100):
        self.urm = urm
        self.w_sparse = np.sum(self.models * self.weights)
        self.w_sparse = similarityMatrixTopK(self.w_sparse, k=top_k).tocsr()

    def get_scores(self, user_id, exclude_seen=True):
        user_profile = self.urm[user_id]
        scores = user_profile.dot(self.w_sparse).toarray().ravel()
        if exclude_seen:
            scores = self.filter_seen(user_id, scores)
        return scores

    def recommend(self, user_id, at=None, exclude_seen=True):
        user_profile = self.urm[user_id]
        if user_profile.nnz == 0 and self.fallback_recommender:
            return self.fallback_recommender.recommend(user_id, at, exclude_seen)
        else:
            scores = self.get_scores(user_id, exclude_seen)
            if self.use_tail_boost:
                scores = self.tb.update_scores(scores)
            ranking = scores.argsort()[::-1]
            return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.urm.indptr[user_id]
        end_pos = self.urm.indptr[user_id + 1]
        user_profile = self.urm.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores


def check_best(bests, item_cf, slim_bpr, slim_enet):
    assert type(bests) == list
    trains, tests, seeds = multiple_splitting()

    cfs = list()
    sbprs = list()
    senets = list()

    for n in range(len(seeds)):
        set_seed(seeds[n])
        top_pop = TopPopRecommender()
        top_pop.fit(trains[n])
        ucb = get_user_cbf(trains[n], generalized=True)
        fb = HybridRecommender([top_pop, ucb], trains[n], merging_type=MergingTechniques.MEDRANK)

        cfs.append(get_item_cf(trains[n], fb=fb, generalized=True))

        slim_bpr = SLIM_BPR(fallback_recommender=fb)
        slim_bpr.fit(trains[n], epochs=300)
        sbprs.append(slim_bpr)

        slim_enet = SLIMElasticNetRecommender(fallback_recommender=fb)
        slim_enet.fit(trains[n])
        senets.append(slim_enet)

    set_seed(42)

    for best in bests:
        top_k = int(best['params']['top_k'])
        w_icf = best['params']['w_icf']
        w_sbpr = best['params']['w_sbpr']
        w_senet = best['params']['w_senet']
        cumulative_MAP = 0
        for n in trange(len(trains)):
            model_hybrid = ModelHybridRecommender([cfs[n].w_sparse, sbprs[n].W, senets[n].W_sparse],
                                                  [w_icf, w_sbpr, w_senet])
            model_hybrid.fit(trains[n], top_k=top_k)
            cumulative_MAP += evaluate(model_hybrid, tests[n], cython=True, verbose=False)['MAP']
        averageMAP = cumulative_MAP / len(trains)
        best['AVG_MAP'] = averageMAP

    bests.sort(key=lambda dic: dic['AVG_MAP'], reverse=True)
    for best in bests:
        print(best)

    return bests


def tuner():
    set_seed(42)
    urm, icm, ucm, target_users = build_all_matrices()
    urm_train, urm_test = train_test_split(urm, SplitType.PROBABILISTIC)

    top_pop = TopPopRecommender()
    top_pop.fit(urm_train)
    user_cbf = get_user_cbf(urm_train, generalized=True)
    hybrid_fb = HybridRecommender([top_pop, user_cbf], urm_train, merging_type=MergingTechniques.MEDRANK)

    item_cf = get_item_cf(urm_train, fb=hybrid_fb, generalized=True)
    slim_bpr = SLIM_BPR(fallback_recommender=hybrid_fb)
    slim_bpr.fit(urm_train, epochs=300)
    slim_enet = SLIMElasticNetRecommender(fallback_recommender=hybrid_fb)
    slim_enet.fit(urm_train)

    pbounds = {
        'top_k': (0, 2000),
        'w_icf': (0, 100),
        'w_sbpr': (0, 1000),
        'w_senet': (0, 1000)
    }

    def to_optimize(top_k, w_icf, w_sbpr, w_senet):
        top_k = int(top_k)
        model_hybrid = ModelHybridRecommender([item_cf.w_sparse, slim_bpr.W, slim_enet.W_sparse],
                                              [w_icf, w_sbpr, w_senet])
        model_hybrid.fit(urm_train, top_k=top_k)
        return evaluate(model_hybrid, urm_test, cython=True, verbose=False)['MAP']

    optimizer = BayesianOptimization(f=to_optimize, pbounds=pbounds)
    optimizer.probe(
        params={'top_k': 977, 'w_icf': 42.82, 'w_sbpr': 535.4, 'w_senet': 52.17},
        lazy=True
    )
    optimizer.maximize(init_points=100, n_iter=200)
    opt_results = optimizer.res
    opt_results.sort(key=lambda dic: dic['target'], reverse=True)
    check_best(opt_results[:10], item_cf, slim_bpr, slim_enet)


if __name__ == '__main__':
    from hybrid import HybridRecommender, MergingTechniques
    tuner()
    exit()
