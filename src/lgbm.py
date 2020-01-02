#!/usr/bin/env python3


import numpy as np
import lightgbm as lgb
from run_utils import build_all_matrices, SplitType, evaluate, export
from sklearn.model_selection import train_test_split
from tqdm import trange


class LGBMRecommender:

    def __init__(self):
        self.params = {
            'learning_rate': 0.003,
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'sub_feature': 0.5,
            'num_leaves': 10,
            'min_data': 50,
            'max_depth': 10,
            'verbose': -1
        }
        self.urm = None
        self.y = None
        self.test = None

    def fit(self, urm, ucm):
        urm = urm.astype(float)
        self.urm = urm
        ucm = ucm.tocsr().astype(float)
        for item_id in trange(urm.shape[1]):
            y = urm[:, item_id].toarray().ravel()
            x = ucm
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            y_test_shape = y_test.shape
            if self.test is None:
                self.test = y_test.reshape(y_test_shape[0], 1)
            else:
                self.test = np.hstack((self.test, y_test.reshape(y_test_shape[0], 1)))
            d_train = lgb.Dataset(x_train, label=y_train)
            #d_test = d_train.create_valid(d_train)
            clf = lgb.train(self.params, d_train, 100, verbose_eval=False)
            y_pred = clf.predict(x_test)
            y_pred_shape = y_pred.shape
            if self.y is None:
                self.y = y_pred.reshape(y_pred_shape[0], 1)
            else:
                self.y = np.hstack((self.y, y_pred.reshape(y_pred_shape[0], 1)))

    def recommend(self, user_id, at=None, exclude_seen=True):
        user_profile = self.urm[user_id]
        ranking = user_profile.argsort()[::-1]
        return ranking[:at]


if __name__ == '__main__':
    EXPORT = False
    np.random.seed(42)
    urm, icm, ucm, target_users = build_all_matrices()

    rec = LGBMRecommender()
    rec.fit(urm, ucm)

    evaluate(rec, rec.test)

