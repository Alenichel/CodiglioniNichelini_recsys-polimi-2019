#!/usr/bin/env python3


import numpy as np
import lightgbm as lgb
import os
from run_utils import set_seed, build_all_matrices, train_test_split, SplitType, evaluate, export, get_seed
from tqdm import trange


class LGBMRecommender:

    def __init__(self):
        self.params = {
            'learning_rate': 0.001,
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'sub_feature': 0.5,
            'num_leaves': 100,
            'min_data': 50,
            'max_depth': 10,
            'verbose': -1,
            #'num_thread': 6,
            'device': 'cpu',
            'max_bin': 200,
            'gpu_use_dp': False
        }
        self.y_pred = None
        self.urm_train = None

    def get_cache_filename(self):
        seed = get_seed()
        urm_train_nnz = self.urm_train.nnz
        return '{seed}_{urm_train_nnz}'.format(seed=seed, urm_train_nnz=urm_train_nnz)

    def fit(self, urm_train, ucm_train, cache=True):
        self.urm_train = urm_train
        cache_dir = 'models/lgbm/'
        cache_file = cache_dir + self.get_cache_filename() + '.npy'
        if cache:
            if os.path.exists(cache_file):
                print('Using cached model')
                self.y_pred = np.load(cache_file, allow_pickle=True)
                return
            else:
                print('{cache_file} not found'.format(cache_file=cache_file))
        urm_train = urm_train.astype(float)
        x_train = ucm_train.tocsr().astype(float)
        n_users, n_items = urm.shape
        self.y_pred = np.zeros(urm.shape, dtype=float)
        for item_id in trange(n_items):
            y_train = urm_train[:, item_id].toarray().ravel()
            d_train = lgb.Dataset(x_train, label=y_train)
            clf = lgb.train(self.params, d_train, 300, verbose_eval=False)
            y_pred = clf.predict(x_train).reshape(n_users)
            self.y_pred[:, item_id] = y_pred
        if cache:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            np.save(cache_file, self.y_pred)
            print('Model cached to file {cache_file}'.format(cache_file=cache_file))

    def recommend(self, user_id, at=None, exclude_seen=True):
        user_profile = self.y_pred[user_id]
        if exclude_seen:
            user_profile = self.filter_seen(user_id, user_profile)
        ranking = user_profile.argsort()[::-1]
        return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.urm_train.indptr[user_id]
        end_pos = self.urm_train.indptr[user_id + 1]
        user_profile = self.urm_train.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores


if __name__ == '__main__':
    EXPORT = False
    set_seed(42)
    urm, icm, ucm, target_users = build_all_matrices()
    # WARNING! This takes only the first thousand users!!!
    urm = urm.tocsr()[3000: 4000, :]
    ucm = ucm.tocsr()[3000: 4000, :]
    urm_train, urm_test = train_test_split(urm, SplitType.PROBABILISTIC)

    rec = LGBMRecommender()
    rec.fit(urm_train, ucm, cache=False)

    evaluate(rec, urm_test)
