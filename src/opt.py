from bayes_opt import BayesianOptimization
from cf import UserCFKNNRecommender, ItemCFKNNRecommender
from cython_modules.SLIM_BPR.SLIM_BPR_CYTHON import SLIM_BPR
from hybrid import HybridRecommender, MergingTechniques
from basic_recommenders import TopPopRecommender
from cbf import ItemCBFKNNRecommender
from run_utils import evaluate, build_all_matrices, train_test_split, SplitType

import numpy as np


def to_optimize(w_cf, w_ucf, w_slim, w_cbf):
    global cf, user_cf, cbf_rec,  slim
    hybrid = HybridRecommender([cf, user_cf, slim, cbf_rec], merging_type=MergingTechniques.WEIGHTS, weights=[w_cf, w_ucf, w_slim, w_cbf])
    return evaluate(hybrid, urm_test)['MAP']


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
    slim.fit(urm_train, epochs=300)

    user_cf = UserCFKNNRecommender(fallback_recommender=tp_rec)
    user_cf.fit(urm_train, top_k=715, shrink=60, normalize=True, similarity='tanimoto')

    cbf_rec = ItemCBFKNNRecommender()
    cbf_rec.fit(urm_train, icm)

    pbounds = {'w_cf': (0, 2), 'w_ucf': (0, 2), 'w_slim': (0, 2), 'w_cbf': (0, 2) }

    optimizer = BayesianOptimization(
        f=to_optimize,
        pbounds=pbounds,
        random_state=1,
        verbose=2
    )

    optimizer.maximize(
        init_points=5,
        n_iter=30,
    )

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))