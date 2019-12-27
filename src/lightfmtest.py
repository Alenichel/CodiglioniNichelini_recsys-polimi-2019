#!/usr/bin/env python3

import numpy as np
from tqdm import trange
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
import matplotlib.pyplot as plt
from run_utils import build_all_matrices, train_test_split, evaluate, export, SplitType
from bayes_opt import BayesianOptimization


class LightFMRecommender:

    def __init__(self):
        self.model = LightFM(loss='warp-kos')
        self.urm = None
        self.n_users = 0
        self.n_items = 0
        self.arange_items = None

    def fit(self, urm, icm, epochs=15, partial=False):
        self.urm = urm.tocsr()
        self.n_users, self.n_items = self.urm.shape
        self.arange_items = np.arange(self.n_items)
        if partial:
            self.model = self.model.fit_partial(self.urm, epochs=epochs, verbose=True)
        else:
            self.model = self.model.fit(self.urm, item_features=icm, epochs=epochs, verbose=True)

    def recommend(self, user_id, at=10, exclude_seen=True):
        scores = self.model.predict(user_id, self.arange_items, item_features=icm)
        if exclude_seen:
            scores = self.__filter_seen(user_id, scores)
        top_items = scores.argsort()[::-1]
        return top_items[:at]

    def __filter_seen(self, user_id, scores):
        start_pos = self.urm.indptr[user_id]
        end_pos = self.urm.indptr[user_id + 1]
        user_profile = self.urm.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores


def rec_round(no_components, learning_rate, item_alpha, user_alpha, max_sampled):
    no_components = int(no_components)
    max_sampled = int(max_sampled)
    model = LightFM(loss='warp',
                    learning_schedule='adagrad',
                    no_components=no_components,
                    learning_rate=learning_rate,
                    item_alpha=item_alpha,
                    user_alpha=user_alpha,
                    max_sampled=max_sampled)
    model.fit(urm_train, user_features=ucm, item_features=icm, epochs=2)
    return precision_at_k(model, urm_test, urm_train, 10, ucm, icm).mean()


if __name__ == '__main__':
    np.random.seed(42)
    EXPORT = False
    urm, icm, ucm, target_users = build_all_matrices()
    if EXPORT:
        urm_train = urm
        urm_test = None
    else:
        urm_train, urm_test = train_test_split(urm, SplitType.PROBABILISTIC)

    pbounds = {
        'no_components': (10, 100),
        'learning_rate': (0.001, 0.1),
        'item_alpha': (0, 1),
        'user_alpha': (0, 1),
        'max_sampled': (1, 100)
    }

    optimizer = BayesianOptimization(f=rec_round, pbounds=pbounds)
    optimizer.maximize(init_points=10, n_iter=200)
    print(optimizer.max)

    '''
    lightfm_rec = LightFMRecommender()
    lightfm_rec.fit(urm_train, icm, epochs=100)
    if EXPORT:
        export(target_users, lightfm_rec)
    else:
        evaluate(lightfm_rec, urm_test)
    '''
