#!/usr/bin/env python3

import numpy as np
import os
import implicit
from run_utils import set_seed, get_seed, build_all_matrices, train_test_split, SplitType, export, evaluate
from bayes_opt import BayesianOptimization


class AlternatingLeastSquare:

    def __init__(self):
        self.urm = None
        self.user_factors = None
        self.item_factors = None

    @staticmethod
    def get_cache_filename(n_factors, regularization, iterations, alpha):
        seed = get_seed()
        return '{seed}_{n_factors}_{regularization}_{iterations}_{alpha}'\
            .format(seed=seed, n_factors=n_factors, regularization=regularization, iterations=iterations, alpha=alpha)

    def fit(self, urm, n_factors=300, regularization=0.15, iterations=30, alpha=24, verbose=True, cache=True):
        self.urm = urm
        cache_dir = 'models/als/'
        cache_file = cache_dir + AlternatingLeastSquare.get_cache_filename(n_factors, regularization, iterations, alpha) + '.npy'
        if cache:
            if os.path.exists(cache_file):
                if verbose:
                    print('ALS: using cached model')
                data = np.load(cache_file, allow_pickle=True)
                self.user_factors = data[0]
                self.item_factors = data[1]
                return
            else:
                print('{cache_file} not found'.format(cache_file=cache_file))
        sparse_item_user = self.urm.T
        # Initialize the als model and fit it using the sparse item-user matrix
        model = implicit.als.AlternatingLeastSquares(factors=n_factors, regularization=regularization, iterations=iterations)
        # Calculate the confidence by multiplying it by alpha.
        data_conf = (sparse_item_user * alpha).astype('double')
        # Fit the model
        model.fit(data_conf, show_progress=verbose)
        # Get the user and item vectors from our trained model
        self.user_factors = model.user_factors
        self.item_factors = model.item_factors
        if cache:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            data = np.array([self.user_factors, self.item_factors])
            np.save(cache_file, data)
            print('Model cached to file {cache_file}'.format(cache_file=cache_file))

    def get_scores(self, user_id, exclude_seen=True):
        scores = np.dot(self.user_factors[user_id], self.item_factors.T)
        return np.squeeze(scores)

    def recommend(self, user_id, at=10):
        user_id = int(user_id)
        expected_ratings = self.get_scores(user_id)
        recommended_items = np.flip(np.argsort(expected_ratings), 0)
        unseen_items_mask = np.in1d(recommended_items, self.urm[user_id].indices,
                                    assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]


def tuner():
    urm, icm, ucm, _ = build_all_matrices()
    urm_train, urm_test = train_test_split(urm, SplitType.PROBABILISTIC)
    pbounds = {'alpha': (10, 50)}

    def rec_round(alpha):
        als = AlternatingLeastSquare()
        als.fit(urm_train, n_factors=896, regularization=99.75, iterations=152, alpha=alpha, cache=False, verbose=False)
        return evaluate(als, urm_test, cython=True, verbose=False)['MAP']

    optimizer = BayesianOptimization(f=rec_round, pbounds=pbounds)
    optimizer.maximize(init_points=30, n_iter=100)
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))
    print(optimizer.max)


if __name__ == '__main__':
    set_seed(42)
    tuner()
    exit()

    EXPORT = False
    urm, icm, ucm, target_users = build_all_matrices()
    if EXPORT:
        urm_train = urm.tocsr()
        urm_test = None
    else:
        urm_train, urm_test = train_test_split(urm, SplitType.PROBABILISTIC)

    als = AlternatingLeastSquare()
    als.fit(urm_train, n_factors=896, regularization=99.75, iterations=152)

    if EXPORT:
        export(target_users, als)
    else:
        evaluate(als, urm_test, cython=False, verbose=True)
