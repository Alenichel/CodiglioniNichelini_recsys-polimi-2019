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
from mf import AlternatingLeastSquare


def to_optimize(n_cluster):
    global user_cbf
    clusters = get_clusters(n_cluster=int(n_cluster))
    top_pop = ClusterizedTopPop()
    top_pop.fit(urm_train, clusters)
    hybrid_fb = HybridRecommender([top_pop, user_cbf], urm_train, merging_type=MergingTechniques.MEDRANK)
    return evaluate(hybrid_fb, urm_test, excluded_users=warm_users, verbose=False)['MAP']

if __name__ == '__main__':
    np.random.seed(42)
    from hybrid import HybridRecommender, MergingTechniques
    from clusterization import get_clusters
    from clusterized_top_pop import ClusterizedTopPop

    urm, icm, ucm, target_users = build_all_matrices()
    urm_train, urm_test = train_test_split(urm, SplitType.PROBABILISTIC)

    # USER CBF
    user_cbf = UserCBFKNNRecommender()
    user_cbf.fit(urm_train, ucm, top_k=496, shrink=0, normalize=False)

    profile_lengths = np.ediff1d(urm_train.indptr)
    warm_users = np.where(profile_lengths != 0)[0]

    pbounds = {
        'n_cluster': (1, 60),
        #'max_iter': (50, 1000)
    }

    optimizer = BayesianOptimization(
        f=to_optimize,
        pbounds=pbounds,
    )

    optimizer.maximize(
        init_points=50,
        n_iter=500,
    )

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))
    print(optimizer.max)
