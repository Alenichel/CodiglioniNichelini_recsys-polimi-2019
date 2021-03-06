#!/usr/bin/env python3

import numpy as np
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.lmf import LogisticMatrixFactorization
from bayes_opt import BayesianOptimization
from run_utils import set_seed, build_all_matrices, train_test_split, SplitType, export, evaluate


class MFALSRecommender:

    def __init__(self):
        self.urm_train = None
        self.model = None

    def __str__(self):
        return 'MFALSRecommender'

    def fit(self, urm_train, factors=100, regularization=0.01, iterations=15, use_gpu=False, verbose=True):
        self.urm_train = urm_train
        self.model = AlternatingLeastSquares(factors=factors, regularization=regularization, iterations=iterations, use_gpu=use_gpu)
        self.model.fit(urm_train.tocoo().T * 24, show_progress=verbose)

    def recommend(self, user_id, at=10, exclude_seen=True):
        recommendations = self.model.recommend(user_id, self.urm_train, at)
        items, _ = zip(*recommendations)
        return items


def als_tuner():
    print('ALS Tuner')

    def rec_round(factors, regularization, iterations):
        factors = int(factors)
        iterations = int(iterations)
        rec = MFALSRecommender()
        rec.fit(urm_train, factors=factors, regularization=regularization, iterations=iterations, verbose=False)
        return evaluate(rec, urm_test, verbose=False)['MAP']

    pbounds = {
        'factors': (50, 150),
        'regularization': (0.001, 0.1),
        'iterations': (1, 100)
    }

    optimizer = BayesianOptimization(f=rec_round, pbounds=pbounds)
    optimizer.maximize(init_points=50, n_iter=300)
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))
    print(optimizer.max)


class MFBPRRecommender:

    def __init__(self):
        self.urm_train = None
        self.model = None

    def __str__(self):
        return 'MFBPRRecommender'

    def fit(self, urm_train, factors=100, learning_rate=0.1, regularization=0.01, iterations=100, use_gpu=False, verbose=True):
        self.urm_train = urm_train
        self.model = BayesianPersonalizedRanking(factors=factors, learning_rate=learning_rate,
                                                 regularization=regularization, iterations=iterations, use_gpu=use_gpu)
        self.model.fit(urm_train.tocoo().T, show_progress=verbose)

    def recommend(self, user_id, at=10, exclude_seen=True):
        recommendations = self.model.recommend(user_id, self.urm_train, at)
        items, _ = zip(*recommendations)
        return items


def bpr_tuner():
    print('BPR Tuner')

    def rec_round(factors, regularization, iterations):
        factors = int(factors)
        iterations = int(iterations)
        bpr = MFBPRRecommender()
        bpr.fit(urm_train, factors=factors, regularization=regularization, iterations=iterations, verbose=False)
        return evaluate(bpr, urm_test, verbose=False)['MAP']

    pbounds = {
        'factors': (50, 150),
        'regularization': (0.001, 0.1),
        'iterations': (1, 300)
    }

    optimizer = BayesianOptimization(f=rec_round, pbounds=pbounds)
    optimizer.maximize(init_points=50, n_iter=300)
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))
    print(optimizer.max)


class LMFRecommender:

    def __init__(self):
        self.urm_train = None
        self.model = None
        self.alpha = 0

    def __str__(self):
        return 'LMFRecommender_{0}'.format(self.alpha)

    def fit(self, urm_train, factors=30, learning_rate=1.0, regularization=0.6, iterations=30, neg_prop=30, alpha=1, verbose=True):
        self.urm_train = urm_train
        self.alpha = alpha
        self.model = LogisticMatrixFactorization(factors=factors,
                                                 learning_rate=learning_rate,
                                                 regularization=regularization,
                                                 iterations=iterations,
                                                 neg_prop=neg_prop)
        self.model.fit(urm_train.tocoo().T * alpha, show_progress=verbose)

    def recommend(self, user_id, at=10, exclude_seen=True):
        recommendations = self.model.recommend(user_id, self.urm_train, at)
        items, _ = zip(*recommendations)
        return items


def lmf_tuner():
    print('LMF Tuner')

    def rec_round(alpha, factors, iterations, regularization):
        factors = int(factors)
        iterations = int(iterations)
        rec = LMFRecommender()
        rec.fit(urm_train, alpha=alpha, factors=factors, iterations=iterations, regularization=regularization, verbose=False)
        return evaluate(rec, urm_test, verbose=False)['MAP']

    pbounds = {
        'factors': (100, 130),
        'regularization': (8, 12),
        'iterations': (1, 100),
        'alpha': (6, 7),
    }

    optimizer = BayesianOptimization(f=rec_round, pbounds=pbounds)
    optimizer.maximize(init_points=50, n_iter=250)
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))
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

    #lmf_tuner()
    #exit()

    rec = LMFRecommender()
    rec.fit(urm_train, alpha=7.5, factors=115, regularization=9.94)

    if EXPORT:
        export(target_users, rec)
    else:
        evaluate(rec, urm_test, verbose=True)
