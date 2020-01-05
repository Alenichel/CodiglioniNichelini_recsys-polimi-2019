#!/usr/bin/env python3

import numpy as np
import similaripy as sim
from run_utils import set_seed, build_all_matrices, train_test_split, SplitType, export, evaluate
from bayes_opt import BayesianOptimization


class SimPyCosineRecommender:

    def __init__(self):
        self.urm_train = None
        self.model = None
        self.recommendations = None

    def fit(self, urm_train, k=4, shrink=34, verbose=True):
        n_users = urm_train.shape[0]
        self.urm_train = urm_train
        self.model = sim.cosine(self.urm_train.T, k=k, shrink=shrink, verbose=verbose)
        self.recommendations = sim.dot_product(self.urm_train, self.model.T, k=10, target_rows=np.arange(n_users),
                                               filter_cols=self.urm_train, verbose=verbose).tocsr()

    def recommend(self, user_id, at=10, exclude_seen=True):
        ranking = np.array(self.recommendations[user_id].todense()).squeeze().argsort()[::-1]
        return ranking[:at]


def tuner():
    pbounds = {
        'k': (0, 100),
        'shrink': (0, 100)
    }

    def rec_round(k, shrink):
        k = int(k)
        shrink = int(shrink)
        rec = SimPyCosineRecommender()
        rec.fit(urm_train, k, shrink, verbose=False)
        return evaluate(rec, urm_test, cython=True, verbose=False)['MAP']

    optimizer = BayesianOptimization(f=rec_round, pbounds=pbounds)
    optimizer.maximize(init_points=30, n_iter=170)
    print(optimizer.max)


if __name__ == '__main__':
    set_seed(42)
    EXPORT = False
    urm, icm, ucm, target_users = build_all_matrices()
    if EXPORT:
        urm_train = urm.tocsr()
        urm_test = None
    else:
        urm_train, urm_test = train_test_split(urm, SplitType.PROBABILISTIC)

    tuner()
    exit()

    rec = SimPyCosineRecommender()
    rec.fit(urm_train)

    if EXPORT:
        export(target_users, rec)
    else:
        evaluate(rec, urm_test)
