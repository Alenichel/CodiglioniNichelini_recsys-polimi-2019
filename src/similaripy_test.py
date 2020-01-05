#!/usr/bin/env python3

from argparse import ArgumentParser
import numpy as np
import similaripy as sim
from run_utils import set_seed, build_all_matrices, train_test_split, SplitType, export, evaluate
from bayes_opt import BayesianOptimization


class SimPyRecommender:

    def __init__(self, similarity):
        assert similarity in [sim.rp3beta, sim.asymmetric_cosine, sim.cosine, sim.dice, sim.jaccard, sim.p3alpha, sim.s_plus, sim.tversky]
        self.similarity = similarity
        self.urm_train = None
        self.model = None
        self.recommendations = None

    def fit(self, urm_train, k=4, shrink=34, verbose=True):
        n_users = urm_train.shape[0]
        self.urm_train = urm_train
        self.model = self.similarity(self.urm_train.T, k=k, shrink=shrink, binary=True, verbose=verbose)
        self.recommendations = sim.dot_product(self.urm_train, self.model.T, k=10, target_rows=np.arange(n_users),
                                               filter_cols=self.urm_train, verbose=verbose).tocsr()

    def recommend(self, user_id, at=10, exclude_seen=True):
        ranking = np.array(self.recommendations[user_id].todense()).squeeze().argsort()[::-1]
        return ranking[:at]


def tuner(similarity):
    pbounds = {
        'k': (1, 200),
        'shrink': (0, 200)
    }

    def rec_round(k, shrink):
        k = int(k)
        shrink = int(shrink)
        rec = SimPyRecommender(similarity)
        rec.fit(urm_train, k, shrink, verbose=False)
        return evaluate(rec, urm_test, cython=True, verbose=False)['MAP']

    optimizer = BayesianOptimization(f=rec_round, pbounds=pbounds)
    optimizer.maximize(init_points=30, n_iter=170)
    print(optimizer.max)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('similarity', choices=['cosine', 'p3alpha', 'rp3beta'])
    parser.add_argument('--tune', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    set_seed(42)
    args = parse_args()
    urm, icm, ucm, target_users = build_all_matrices()
    urm_train, urm_test = train_test_split(urm, SplitType.PROBABILISTIC)

    similarity = None
    if args.similarity == 'cosine':
        similarity = sim.cosine
    elif args.similarity == 'p3alpha':
        similarity = sim.p3alpha
    elif args.similarity == 'rp3beta':
        similarity = sim.rp3beta

    if args.tune:
        tuner(similarity)
    else:
        rec = SimPyRecommender(similarity)
        rec.fit(urm_train)
        evaluate(rec, urm_test)
