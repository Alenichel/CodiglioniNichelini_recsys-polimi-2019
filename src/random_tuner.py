#!/usr/bin/env python3

import numpy as np
from run_utils import build_all_matrices, train_test_split, SplitType, evaluate
from cf import ItemCFKNNRecommender, UserCFKNNRecommender
from cbf import ItemCBFKNNRecommender, UserCBFKNNRecommender
from skopt.space import Integer, Categorical
from tqdm import trange


class RandomTuner:

    def __init__(self, recommender_class, urm, icm, fit_params_ranges):
        self.recommender_class = recommender_class
        self.needs_icm = self.recommender_class in [ItemCBFKNNRecommender, UserCBFKNNRecommender]
        self.urm = urm
        self.icm = icm
        self.fit_params_ranges = fit_params_ranges

    def tune(self, n_rounds=100, n_rounds_per_rec=10):
        results = list()
        for _ in trange(n_rounds, desc='Rounds'):
            fit_params = dict()
            for k in self.fit_params_ranges.keys():
                param_range = self.fit_params_ranges[k]
                if isinstance(param_range, Integer):
                    fit_params[k] = np.random.randint(param_range.low, param_range.high)
                elif isinstance(param_range, Categorical):
                    fit_params[k] = np.random.choice(param_range.categories)
            print(fit_params)
            round_maps = []
            for _ in trange(n_rounds_per_rec, desc='Rec rounds'):
                urm_train, urm_test = train_test_split(self.urm, split_type=SplitType.LOO_CYTHON)
                recommender = self.recommender_class()
                if self.needs_icm:
                    recommender.fit(urm_train, self.icm, **fit_params)
                else:
                    recommender.fit(urm_train, **fit_params)
                round_maps.append(evaluate(recommender, urm_test, cython=True)['MAP'])
            result_map = sum(round_maps) / n_rounds_per_rec
            results.append((fit_params, result_map))
            print(sorted(results, key=lambda x: x[1])[::-1])
        return sorted(results, key=lambda x: x[1])[::-1]


if __name__ == '__main__':
    fit_params_ranges = {
        'top_k': Integer(1, 1000),
        'shrink': Integer(0, 1000),
        'similarity': Categorical(['cosine', 'tanimoto']),
        'normalize': Categorical([True, False])
    }
    urm, icm, ucm, _ = build_all_matrices()
    tuner = RandomTuner(UserCBFKNNRecommender, urm, ucm, fit_params_ranges)
    results = tuner.tune(n_rounds=10, n_rounds_per_rec=3)
    for r in results[::-1]:
        print(r)
