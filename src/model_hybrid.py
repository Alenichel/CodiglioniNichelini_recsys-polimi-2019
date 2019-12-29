#!/usr/bin/env python3

import numpy as np
from Base.Recommender_utils import similarityMatrixTopK
from run_utils import build_all_matrices, train_test_split, SplitType, export, evaluate
from basic_recommenders import TopPopRecommender
from cf import ItemCFKNNRecommender
from cbf import UserCBFKNNRecommender
from cython_modules.SLIM_BPR.SLIM_BPR_CYTHON import SLIM_BPR
from slim_elasticnet import SLIMElasticNetRecommender
from bayes_opt import BayesianOptimization


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


def to_optimize(top_k, w_icf, w_sbpr, w_senet):
    top_k = int(top_k)
    model_hybrid = ModelHybridRecommender([item_cf.w_sparse, slim_bpr.W, slim_enet.W_sparse], [w_icf, w_sbpr, w_senet])
    model_hybrid.fit(urm_train, top_k=top_k)
    return evaluate(model_hybrid, urm_test, cython=True, verbose=False)['MAP']


if __name__ == '__main__':
    from hybrid import HybridRecommender, MergingTechniques
    np.random.seed(42)
    EXPORT = False
    urm, icm, ucm, target_users = build_all_matrices()
    if EXPORT:
        urm_train = urm.tocsr()
        urm_test = None
    else:
        urm_train, urm_test = train_test_split(urm, SplitType.PROBABILISTIC)

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
    slim_enet.fit(urm_train)
    # MODEL HYBRID
    model_hybrid = ModelHybridRecommender([item_cf.w_sparse, slim_bpr.W, slim_enet.W_sparse], [42.82, 535.4, 52.17], fallback_recommender=hybrid_fb)
    model_hybrid.fit(urm_train, top_k=977)

    if EXPORT:
        export(target_users, model_hybrid)
    else:
        evaluate(model_hybrid, urm_test)
    exit()

    pbounds = {
        'top_k': (0, 1000),
        'w_icf': (0, 100),
        'w_sbpr': (0, 1000),
        'w_senet': (0, 1000)
    }

    optimizer = BayesianOptimization(
        f=to_optimize,
        pbounds=pbounds
    )

    optimizer.maximize(
        init_points=10,
        n_iter=100
    )

    print(optimizer.max)
