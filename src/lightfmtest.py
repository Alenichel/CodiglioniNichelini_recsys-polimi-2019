#!/usr/bin/env python3

import numpy as np
from lightfm import LightFM
from run_utils import build_all_matrices, train_test_split, evaluate, export, SplitType


class LightFMRecommender:

    def __init__(self):
        self.model = LightFM(loss='warp-kos')
        self.urm = None
        self.n_users = 0
        self.n_items = 0
        self.arange_items = None

    def fit(self, urm, epochs=15, partial=False):
        self.urm = urm.tocsr()
        self.n_users, self.n_items = self.urm.shape
        self.arange_items = np.arange(self.n_items)
        if partial:
            self.model.fit_partial(self.urm, epochs=epochs, verbose=True)
        else:
            self.model.fit(self.urm, epochs=epochs, verbose=True)

    def recommend(self, user_id, at=10, exclude_seen=True):
        scores = self.model.predict(user_id, self.arange_items)
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


if __name__ == '__main__':
    EXPORT = False
    urm, icm, target_users = build_all_matrices()
    if EXPORT:
        urm_train = urm
        urm_test = None
    else:
        urm_train, urm_test = train_test_split(urm, SplitType.LOO)
    lightfm_rec = LightFMRecommender()
    lightfm_rec.fit(urm_train, epochs=100)
    if EXPORT:
        export(target_users, lightfm_rec)
    else:
        evaluate(lightfm_rec, urm_test)
