#!/usr/bin/env python3

import numpy as np
from bayes_opt import BayesianOptimization
from cf import UserCFKNNRecommender, ItemCFKNNRecommender
from cython_modules.SLIM_BPR.SLIM_BPR_CYTHON import SLIM_BPR
from slim_elasticnet import SLIMElasticNetRecommender
from hybrid import HybridRecommender, MergingTechniques
from basic_recommenders import TopPopRecommender
from cbf import ItemCBFKNNRecommender, UserCBFKNNRecommender
from run_utils import evaluate, build_all_matrices, train_test_split, SplitType


def to_optimize(w_cf, w_ucf, w_slim, w_slim_el, w_cbf):
    global item_cf, user_cf, slim_bpr, slim_elasticnet, item_cbf
    hybrid = HybridRecommender([item_cf, user_cf, slim_bpr, slim_elasticnet, item_cbf],
                               merging_type=MergingTechniques.WEIGHTS,
                               weights=[w_cf, w_ucf, w_slim, w_slim_el, w_cbf])
    return evaluate(hybrid, urm_test, cython=True, verbose=False)['MAP']


if __name__ == '__main__':
    np.random.seed(42)

    EXPORT = False
    urm, icm, ucm, target_users = build_all_matrices()
    if EXPORT:
        urm_train = urm.tocsr()
        urm_test = None
    else:
        urm_train, urm_test = train_test_split(urm, SplitType.PROBABILISTIC)
    n_users, n_items = urm_train.shape

    top_pop = TopPopRecommender()
    top_pop.fit(urm_train)

    user_cbf = UserCBFKNNRecommender()
    user_cbf.fit(urm_train, ucm, top_k=496, shrink=0, normalize=False)

    hybrid_fb = HybridRecommender([top_pop, user_cbf], merging_type=MergingTechniques.RR)

    item_cf = ItemCFKNNRecommender(fallback_recommender=hybrid_fb)
    item_cf.fit(urm_train, top_k=5, shrink=20, similarity='tanimoto')

    slim_bpr = SLIM_BPR(fallback_recommender=hybrid_fb)
    slim_bpr.fit(urm_train, epochs=300)

    slim_elasticnet = SLIMElasticNetRecommender()
    slim_elasticnet.fit(urm_train)

    user_cf = UserCFKNNRecommender(fallback_recommender=hybrid_fb)
    user_cf.fit(urm_train, top_k=715, shrink=60, normalize=True, similarity='tanimoto')

    item_cbf = ItemCBFKNNRecommender()
    item_cbf.fit(urm_train, icm, 417, 0, normalize=True)

    bounds = (0, 1)
    pbounds = {'w_cf': bounds, 'w_ucf': bounds, 'w_slim': bounds, 'w_slim_el': bounds, 'w_cbf': bounds}

    optimizer = BayesianOptimization(
        f=to_optimize,
        pbounds=pbounds,
    )

    optimizer.maximize(
        init_points=10,
        n_iter=300,
    )

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))
    print(optimizer.max)
