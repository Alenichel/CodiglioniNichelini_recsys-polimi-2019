#!/usr/bin/env python3


import numpy as np
import scipy.sparse as sps
import lightgbm as lgb
from os.path import exists
from run_utils import build_all_matrices, train_test_split, SplitType, evaluate, export
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
            'verbose': -1,
            #'num_thread': 6,
            'device': 'cpu',
            'max_bin': 15,
            'gpu_use_dp': False
        }
        self.y_pred = None

    @staticmethod
    def get_cache_filename():
        seed = np.random.get_state()[1][0]
        return '{seed}'.format(seed=seed)

    def fit(self, urm_train, ucm_train, ucm_test, cache=True):
        cache_file = 'models/lgbm/' + LGBMRecommender.get_cache_filename() + '.npy'
        if cache:
            if exists(cache_file):
                print('Using cached model')
                self.y_pred = np.load(cache_file, allow_pickle=True)
                return
            else:
                print('{cache_file} not found'.format(cache_file=cache_file))
        urm_train = urm_train.astype(float)
        x_train = ucm_train.tocsr().astype(float)
        x_test = ucm_test.tocsr().astype(float)
        n_users, n_items = urm.shape
        for item_id in trange(n_items):
            y_train = urm_train[:, item_id].toarray().ravel()
            d_train = lgb.Dataset(x_train, label=y_train)
            clf = lgb.train(self.params, d_train, 100, verbose_eval=False)
            y_pred = clf.predict(x_test)
            y_pred_shape = y_pred.shape
            if self.y_pred is None:
                self.y_pred = y_pred.reshape(y_pred_shape[0], 1)
            else:
                self.y_pred = np.hstack((self.y_pred, y_pred.reshape(y_pred_shape[0], 1)))
        if cache:
            np.save(cache_file, self.y_pred)
            print('Model cached to file {cache_file}'.format(cache_file=cache_file))

    def recommend(self, user_id, at=None, exclude_seen=True):
        user_profile = self.y_pred[user_id]
        ranking = user_profile.argsort()[::-1]
        return ranking[:at]


if __name__ == '__main__':
    EXPORT = False
    np.random.seed(42)
    urm, icm, ucm, target_users = build_all_matrices()
    urm_train, urm_test = train_test_split(urm, SplitType.PROBABILISTIC)
    ucm_train, ucm_test = train_test_split(ucm, SplitType.PROBABILISTIC)

    rec = LGBMRecommender()
    rec.fit(urm_train, ucm_train, ucm_test)

    evaluate(rec, urm_test)
