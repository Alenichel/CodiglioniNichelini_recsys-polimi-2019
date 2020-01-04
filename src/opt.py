#!/usr/bin/env python3

import numpy as np
from bayes_opt import BayesianOptimization
from cf import UserCFKNNRecommender, ItemCFKNNRecommender
from cython_modules.SLIM_BPR.SLIM_BPR_CYTHON import SLIM_BPR
from model_hybrid import ModelHybridRecommender
from slim_elasticnet import SLIMElasticNetRecommender
from hybrid import HybridRecommender, MergingTechniques
from basic_recommenders import TopPopRecommender
from cbf import ItemCBFKNNRecommender, UserCBFKNNRecommender
from run_utils import evaluate, build_all_matrices, train_test_split, SplitType, set_seed, user_segmenter
from mf import AlternatingLeastSquare

def to_optimize(w_model, w_ucf, w_icbf, w_als):
    global to_exclude, hybrid_fb, model_hybrid, user_cf, als, item_cbf
    hybrid = HybridRecommender([model_hybrid, user_cf, item_cbf, als],
                               urm_train,
                               merging_type=MergingTechniques.WEIGHTS,
                               weights=[w_model, w_ucf, w_icbf, w_als],
                               fallback_recommender=hybrid_fb)
    return evaluate(hybrid, urm_test, excluded_users=to_exclude, verbose=False)['MAP']


if __name__ == '__main__':
    set_seed(42)
    from hybrid import HybridRecommender, MergingTechniques
    from clusterization import get_clusters
    from clusterized_top_pop import ClusterizedTopPop

    urm, icm, ucm, target_users = build_all_matrices()
    urm_train, urm_test = train_test_split(urm, SplitType.PROBABILISTIC)

    groups, _ = user_segmenter(urm_train, n_groups=10)
    to_exclude = groups[1].not_in_group

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
    slim_enet.fit(urm_train, cache=True)
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
    als.fit(urm_train, n_factors=896, regularization=99.75, iterations=152, cache=True)

    hybrid = HybridRecommender([model_hybrid, user_cf, item_cbf, als],
                               urm_train,
                               merging_type=MergingTechniques.WEIGHTS,
                               weights=[0.4767, 2.199, 2.604, 7.085],
                               fallback_recommender=hybrid_fb)




    for n in range(10):
        to_exclude = groups[n].not_in_group
        evaluate(hybrid, urm_test, verbose=True, excluded_users=to_exclude)

    exit()

    pbounds = {
        'w_model': (0.5, 1),
        'w_ucf': (2, 2.5),
        'w_icbf': (2.7, 3.2),
        'w_als': (6.5, 8),
    }

    optimizer = BayesianOptimization(
     f=to_optimize,
     pbounds=pbounds,
    )

    optimizer.maximize(
     init_points=40,
     n_iter=100,
    )

    for i, res in enumerate(optimizer.res):
     print("Iteration {}: \n\t{}".format(i, res))
    print(optimizer.max)
