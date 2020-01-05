#!/usr/bin/env python3

import numpy as np
import similaripy as sim
from run_utils import set_seed, build_all_matrices, train_test_split, SplitType, export, evaluate


class SimPyRecommender:

    def __init__(self):
        self.urm_train = None
        self.model = None
        self.recommendations = None

    def fit(self, urm_train, k=4, shrink=34):
        n_users = urm_train.shape[0]
        self.urm_train = urm_train
        self.model = sim.cosine(self.urm_train.T, k=k, shrink=shrink)
        self.recommendations = sim.dot_product(self.urm_train, self.model.T, k=10, target_rows=np.arange(n_users),
                                               filter_cols=self.urm_train).tocsr()

    def recommend(self, user_id, at=10, exclude_seen=True):
        ranking = np.array(self.recommendations[user_id].todense()).squeeze().argsort()[::-1]
        return ranking[:at]


if __name__ == '__main__':
    set_seed(42)
    EXPORT = False
    urm, icm, ucm, target_users = build_all_matrices()
    if EXPORT:
        urm_train = urm.tocsr()
        urm_test = None
    else:
        urm_train, urm_test = train_test_split(urm, SplitType.PROBABILISTIC)

    rec = SimPyRecommender()
    rec.fit(urm_train)

    if EXPORT:
        export(target_users, rec)
    else:
        evaluate(rec, urm_test)
