#!/usr/bin/env python3

import numpy as np
from run_utils import build_all_matrices, train_test_split, SplitType, evaluate
from cf import ItemCFKNNRecommender
from skopt.space import Integer, Categorical
from tqdm import trange


class RandomTuner:

    def __init__(self, recommender_class, urm, fit_params_ranges):
        self.recommender_class = recommender_class
        self.urm = urm
        self.fit_params_ranges = fit_params_ranges

    def tune(self, n_rounds=100):
        results = list()
        for _ in trange(n_rounds):
            urm_train, urm_test = train_test_split(self.urm, split_type=SplitType.LOO_CYTHON)
            recommender = self.recommender_class()
            fit_params = dict()
            for k in self.fit_params_ranges.keys():
                param_range = self.fit_params_ranges[k]
                if isinstance(param_range, Integer):
                    fit_params[k] = np.random.randint(param_range.low, param_range.high)
                elif isinstance(param_range, Categorical):
                    fit_params[k] = np.random.choice(param_range.categories)
            print(fit_params)
            recommender.fit(urm_train, **fit_params)
            map = evaluate(recommender, urm_test, cython=True)['MAP']
            results.append((fit_params, map))
        return sorted(results, key=lambda x: x[1])[::-1]


if __name__ == '__main__':
    fit_params_ranges = {
        'top_k': Integer(1, 1000),
        'shrink': Integer(0, 1000),
        'similarity': Categorical(['cosine', 'tanimoto']),
        'normalize': Categorical([True, False])
    }
    urm, _, _ = build_all_matrices()
    tuner = RandomTuner(ItemCFKNNRecommender, urm, fit_params_ranges)
    results = tuner.tune(3)
    print(results)
