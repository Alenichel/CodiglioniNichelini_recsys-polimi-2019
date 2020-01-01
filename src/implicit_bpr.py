#!/usr/bin/env python3

import numpy as np
from implicit.bpr import BayesianPersonalizedRanking
from bayes_opt import BayesianOptimization
from run_utils import build_all_matrices, train_test_split, SplitType, export, evaluate


class BPRRecommender:

    def __init__(self):
        self.urm_train = None
        self.model = None

    def fit(self, urm_train, factors=100, learning_rate=0.1, regularization=0.01, iterations=100, verbose=False):
        self.urm_train = urm_train
        self.model = BayesianPersonalizedRanking(factors=factors, learning_rate=learning_rate, regularization=regularization, iterations=iterations)
        self.model.fit(urm_train.tocoo().T, show_progress=verbose)

    def recommend(self, user_id, at=10):
        recommendations = self.model.recommend(user_id, self.urm_train, at)
        items, _ = zip(*recommendations)
        return items


def tuner():
    urm, icm, ucm, _ = build_all_matrices()
    urm_train, urm_test = train_test_split(urm, SplitType.PROBABILISTIC)

    def rec_round(factors, learning_rate, regularization, iterations):
        factors = int(factors)
        iterations = int(iterations)
        bpr = BPRRecommender()
        bpr.fit(urm_train, factors, learning_rate, regularization, iterations)
        return evaluate(bpr, urm_test, verbose=False)['MAP']

    pbounds = {
        'factors': (50, 150),
        'learning_rate': (0.001, 0.1),
        'regularization': (0.001, 0.1),
        'iterations': (1, 300)
    }

    optimizer = BayesianOptimization(f=rec_round, pbounds=pbounds)
    optimizer.maximize(init_points=50, n_iter=300)
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))
    print(optimizer.max)


if __name__ == '__main__':
    np.random.seed(42)
    tuner()
    exit()

    EXPORT = False
    urm, icm, ucm, target_users = build_all_matrices()
    if EXPORT:
        urm_train = urm.tocsr()
        urm_test = None
    else:
        urm_train, urm_test = train_test_split(urm, SplitType.PROBABILISTIC)

    '''bpr = BPRRecommender()
    bpr.fit(urm_train, n_factors=896, regularization=99.75, iterations=152)

    if EXPORT:
        export(target_users, als)
    else:
        evaluate(als, urm_test, cython=False, verbose=True)'''
