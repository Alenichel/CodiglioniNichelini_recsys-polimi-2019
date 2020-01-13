#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import trange
from run_utils import set_seed, build_urm, train_test_split, build_age_ucm, build_region_ucm, get_seed, evaluate


class LGBMRecommender:

    def __init__(self, urm_train, ucm_age, ucm_region):
        self.urm_train = urm_train.tocsr()
        self.n_users, self.n_items = urm_train.shape
        self.profile_length = np.ediff1d(urm_train.tocsr().indptr)
        cache_dir = 'models/lgbm/'
        cache_filename = cache_dir + self.get_cache_filename() + '.csv'
        if not os.path.exists(cache_filename):
            data = {
                #'user_id': list(range(self.n_users)),
                'profile_length': [self.profile_length[user_id] for user_id in range(self.n_users)],
                'user_age': [LGBMRecommender.__get_if_present(ucm_age, user_id) for user_id in range(self.n_users)],
                'user_region': [LGBMRecommender.__get_if_present(ucm_region, user_id) for user_id in range(self.n_users)],
            }
            self.df = pd.DataFrame(data, dtype=int)
            self.df.to_csv(cache_filename)
        else:
            self.df = pd.read_csv(cache_filename)
            self.df = self.df.drop(['Unnamed: 0'], axis=1)
        self.params = {
            'learning_rate': 0.01,
            'boosting_type': 'gbdt',
            'objective': 'binary',
            # 'metric': 'binary_logloss',
            # 'sub_feature': 0.5,
            # 'num_leaves': 100,
            # 'min_data': 50,
            # 'max_depth': 10,
            'verbose': -1,
            # 'num_thread': 6,
            # 'device': 'cpu',
            # 'max_bin': 200,
            # 'gpu_use_dp': False
        }
        self.y_pred = np.zeros(urm_train.shape, dtype=float)

    def get_cache_filename(self):
        seed = get_seed()
        urm_train_nnz = self.urm_train.nnz
        return '{seed}_{urm_train_nnz}_new_new' \
            .format(seed=seed, urm_train_nnz=urm_train_nnz)

    @staticmethod
    def __get_if_present(csr, row_index):
        values = csr[row_index].indices
        length = len(values)
        # return values[0] if length > 0 else np.nan
        return values[length - 1] if length > 0 else 42424242

    def fit(self):
        feature_names = [c for c in self.df.columns if not c.endswith('_id')]
        cat_features = [c for c in self.df.columns if not c.endswith('_id')]
        for item_id in trange(self.n_items):
            y = self.urm_train[:, item_id].toarray().ravel()
            train_dataset = lgb.Dataset(self.df,
                                        label=y,
                                        feature_name=feature_names,
                                        categorical_feature=cat_features,
                                        free_raw_data=False)
            model = lgb.train(self.params, train_set=train_dataset, feature_name=feature_names, categorical_feature=cat_features)
            preds = model.predict(self.df)
            self.y_pred[:, item_id] = preds

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
    set_seed(42)
    urm = build_urm()
    urm_train, urm_test = train_test_split(urm)
    urm_train = urm_train.tocoo()
    ucm_age = build_age_ucm(urm_train.shape[0])
    ucm_region = build_region_ucm(urm_train.shape[0])

    rec = LGBMRecommender(urm_train, ucm_age, ucm_region)
    rec.fit()

    evaluate(rec, urm_test)
