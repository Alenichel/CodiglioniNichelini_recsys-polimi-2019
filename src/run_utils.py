#!/usr/bin/env python3

from enum import Enum
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm, trange
from cython_modules.leave_one_out import train_test_loo_split as __train_test_loo_split_cython
from csv_utils import load_csv, export_csv
from evaluation import evaluate_algorithm


class DataFiles:
    TRAIN = 'data/data_train.csv'
    TARGET_USERS_TEST = 'data/data_target_users_test.csv'
    ICM_ASSET = 'data/data_ICM_asset.csv'
    ICM_PRICE = 'data/data_ICM_price.csv'
    ICM_SUBCLASS = 'data/data_ICM_sub_class.csv'
    UCM_AGE = 'data/data_UCM_age.csv'
    UCM_REGION = 'data/data_UCM_region.csv'


class SplitType(Enum):
    PROBABILISTIC = 1
    LOO = 2
    LOO_CYTHON = 3


def build_urm():
    urm_data = load_csv(DataFiles.TRAIN)
    urm_data = [[int(row[i]) if i <= 1 else int(float(row[i])) for i in range(len(row))] for row in urm_data]
    users, items, ratings = map(np.array, zip(*urm_data))
    return sps.csr_matrix((ratings, (users, items)))


def build_icm(n_items):
    # PRICE
    price_icm_items, _, price_icm_values = __load_icm_csv(DataFiles.ICM_PRICE, third_type=float)
    price_icm_values = __encode_values(price_icm_values)
    n_features = max(price_icm_values) + 1
    shape = (n_items, n_features)
    ones = np.ones(len(price_icm_values))
    price_icm = sps.csr_matrix((ones, (price_icm_items, price_icm_values)), shape=shape, dtype=int)

    # ASSET
    asset_icm_items, _, asset_icm_values = __load_icm_csv(DataFiles.ICM_ASSET, third_type=float)
    asset_icm_values += 1
    asset_icm_values = __encode_values(asset_icm_values)
    n_features = max(asset_icm_values) + 1
    shape = (n_items, n_features)
    ones = np.ones(len(asset_icm_values))
    asset_icm = sps.csr_matrix((ones, (asset_icm_items, asset_icm_values)), shape=shape, dtype=int)

    # SUBCLASS
    subclass_icm_items, subclass_icm_features, subclass_icm_values = __load_icm_csv(DataFiles.ICM_SUBCLASS, third_type=float)
    n_features = max(subclass_icm_features) + 1
    shape = (n_items, n_features)
    subclass_icm = sps.csr_matrix((subclass_icm_values, (subclass_icm_items, subclass_icm_features)), shape=shape, dtype=int)

    return sps.hstack((price_icm, asset_icm, subclass_icm)).tocsr()


def build_target_users():
    target_users = load_csv(DataFiles.TARGET_USERS_TEST)
    return [int(x[0]) for x in target_users]


def build_all_matrices():
    urm = build_urm()
    n_items = urm.shape[1]
    icm = build_icm(n_items)
    target_users = build_target_users()
    return urm, icm, target_users


def train_test_split(urm, split_type=SplitType.LOO, split=0.8):
    if split_type == SplitType.PROBABILISTIC:
        return __train_test_split(urm, split)
    elif split_type == SplitType.LOO:
        return __train_test_loo_split(urm)
    elif split_type == SplitType.LOO_CYTHON:
        return __train_test_loo_split_cython(urm)


def evaluate(recommender, urm_test, excluded_users=[], cython=False):
    if cython:
        print('Ignoring argument excluded_users')
        from cython_modules.evaluation import evaluate_cython
        print('Using Cython evaluation')
        return evaluate_cython(recommender, urm_test)
    else:
        return evaluate_algorithm(recommender, urm_test, excluded_users=excluded_users)


def export(target_users, recommender):
    print('Exporting recommendations...')
    data = list()
    for u_id in tqdm(target_users, desc='Export'):
        data.append((u_id, recommender.recommend(u_id, at=10)))
    export_csv(('user_id', 'item_list'), data)
    print('OK')


def __train_test_split(urm, split=0.8):
    print('Using probabilistic splitting ({0:.2f}/{1:.2f})'.format(split, 1-split))
    urm = urm.tocoo()
    num_interactions = urm.nnz
    shape = urm.shape
    train_mask = np.random.choice([True, False], num_interactions, p=[split, 1-split])
    urm_train = sps.coo_matrix((urm.data[train_mask], (urm.row[train_mask], urm.col[train_mask])), shape=shape)
    urm_train = urm_train.tocsr()
    test_mask = np.logical_not(train_mask)
    urm_test = sps.coo_matrix((urm.data[test_mask], (urm.row[test_mask], urm.col[test_mask])), shape=shape)
    urm_test = urm_test.tocsr()
    return urm_train, urm_test


def __train_test_loo_split(urm):
    print('Using LeaveOneOut')
    urm = urm.tocsr()
    num_users = urm.shape[0]
    num_items = urm.shape[1]
    urm_train = urm.copy()
    urm_test = sps.lil_matrix((num_users, num_items), dtype=int)
    for user_id in trange(num_users, desc='LeaveOneOut'):
        start_pos = urm_train.indptr[user_id]
        end_pos = urm_train.indptr[user_id + 1]
        user_profile = urm_train.indices[start_pos:end_pos]
        if user_profile.size > 0:
            item_id = np.random.choice(user_profile, 1)
            urm_train[user_id, item_id] = 0
            urm_test[user_id, item_id] = 1
    urm_test = sps.csr_matrix(urm_test, dtype=int, shape=urm.shape)
    urm_train.eliminate_zeros()
    urm_test.eliminate_zeros()
    return urm_train, urm_test


def __load_icm_csv(filename, third_type):
    data = load_csv(filename)
    data = [[int(row[i]) if i <= 1 else third_type(row[i]) for i in range(len(row))] for row in data]
    items, features, values = map(np.array, zip(*data))
    return items, features, values


def __encode_values(values):
    le = LabelEncoder()
    le.fit(values)
    return le.transform(values)
