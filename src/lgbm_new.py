#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import trange
from run_utils import set_seed, build_urm, train_test_split, build_age_ucm, build_region_ucm, build_price_icm, \
    build_asset_icm, build_subclass_icm, get_seed, evaluate


class LGBMRecommender:

    def __init__(self, urm_train, ucm_age, ucm_region, icm_price, icm_asset, icm_subclass):
        self.urm_train = urm_train.tocsr()
        n_users, n_items = urm_train.shape
        self.profile_length = np.ediff1d(urm_train.tocsr().indptr)
        cache_dir = 'models/lgbm/'
        cache_filename = cache_dir + self.get_cache_filename() + '.csv'
        if not os.path.exists(cache_filename):
            data = {
                'user_id': list(),
                'item_id': list(),
                'profile_length': list(),
                'item_popularity': list(),
                'user_age': list(),
                'user_region': list(),
                'item_price': list(),
                'item_asset': list(),
                'item_subclass': list(),
                'target': list(),
            }
            item_popularity = np.array(urm_train.tocsr().sum(axis=0)).squeeze()
            n_interactions = len(urm_train.data)
            for i in trange(n_interactions):
                user_id = urm_train.row[i]
                item_id = urm_train.col[i]
                data['user_id'].append(user_id)
                data['item_id'].append(item_id)
                data['profile_length'].append(self.profile_length[user_id])
                data['item_popularity'].append(item_popularity[item_id])
                data['user_age'].append(LGBMRecommender.__get_if_present(ucm_age, user_id))
                data['user_region'].append(LGBMRecommender.__get_if_present(ucm_region, user_id))
                data['item_price'].append(LGBMRecommender.__get_if_present(icm_price, item_id))
                data['item_asset'].append(LGBMRecommender.__get_if_present(icm_asset, item_id))
                data['item_subclass'].append(LGBMRecommender.__get_if_present(icm_subclass, item_id))
                data['target'].append(1)
            for _ in trange(n_interactions):
                while True:
                    user_id = np.random.randint(n_users)
                    item_id = np.random.randint(n_items)
                    if self.urm_train[user_id, item_id] == 0:
                        break
                data['user_id'].append(user_id)
                data['item_id'].append(item_id)
                data['profile_length'].append(self.profile_length[user_id])
                data['item_popularity'].append(item_popularity[item_id])
                data['user_age'].append(LGBMRecommender.__get_if_present(ucm_age, user_id))
                data['user_region'].append(LGBMRecommender.__get_if_present(ucm_region, user_id))
                data['item_price'].append(LGBMRecommender.__get_if_present(icm_price, item_id))
                data['item_asset'].append(LGBMRecommender.__get_if_present(icm_asset, item_id))
                data['item_subclass'].append(LGBMRecommender.__get_if_present(icm_subclass, item_id))
                data['target'].append(0)
            for k in data:
                assert len(data[k]) == n_interactions * 2
            self.df = pd.DataFrame(data, dtype=int) #.sample(frac=1).reset_index(drop=True)
            self.df.to_csv(cache_filename)
        else:
            self.df = pd.read_csv(cache_filename)
            self.df = self.df.drop(['Unnamed: 0'], axis=1)
        self.x = None
        self.dataset = None
        self.model = None
        self.params = {
            'learning_rate': 0.01,
            'boosting_type': 'dart',
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
        dummy_list = list(range(n_items))
        self.recommend_data = {
            'item_id': dummy_list,
            'profile_length': dummy_list,
            'item_popularity': dummy_list,
            'user_age': dummy_list,
            'user_region': dummy_list,
            'item_price': dummy_list,
            'item_asset': dummy_list,
            'item_subclass': dummy_list,
        }

    @staticmethod
    def __get_if_present(csr, row_index):
        values = csr[row_index].indices
        length = len(values)
        # return values[0] if length > 0 else np.nan
        return values[length - 1] if length > 0 else -1

    def get_cache_filename(self):
        seed = get_seed()
        urm_train_nnz = self.urm_train.nnz
        return '{seed}_{urm_train_nnz}' \
            .format(seed=seed, urm_train_nnz=urm_train_nnz)

    def fit(self):
        self.x = self.df.drop(['target'], axis=1)
        self.dataset = lgb.Dataset(self.x, label=self.df.target)
        self.model = lgb.train(params=self.params, train_set=self.dataset)

    def recommend(self, user_id, at=None, exclude_seen=True):
        n_items = self.urm_train.shape[1]
        self.recommend_data['user_id'] = [user_id for _ in range(n_items)]
        recommend_df = pd.DataFrame(self.recommend_data, dtype=int)
        scores = self.model.predict(recommend_df)
        if exclude_seen:
            scores = self.filter_seen(user_id, scores)
        ranking = scores.argsort()[::-1]
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
    icm_price = build_price_icm(urm_train.shape[1])
    icm_asset = build_asset_icm(urm_train.shape[1])
    icm_subclass = build_subclass_icm(urm_train.shape[1])

    rec = LGBMRecommender(urm_train, ucm_age, ucm_region, icm_price, icm_asset, icm_subclass)
    rec.fit()

    evaluate(rec, urm_test)
