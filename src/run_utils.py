#!/usr/bin/env python3

import os
from enum import Enum
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm, trange
from cython_modules.leave_one_out import train_test_loo_split as __train_test_loo_split_cython
from csv_utils import load_csv, export_csv
from multiprocessing import Pool
from collections import namedtuple


class DataFiles:
    TRAIN = 'data/data_train.csv'
    TARGET_USERS_TEST = 'data/data_target_users_test.csv'
    ICM_ASSET = 'data/data_ICM_asset.csv'
    ICM_PRICE = 'data/data_ICM_price.csv'
    ICM_SUBCLASS = 'data/data_ICM_sub_class.csv'
    UCM_AGE = 'data/data_UCM_age.csv'
    UCM_REGION = 'data/data_UCM_region.csv'
    CLUSTERS = 'data/user_clustered.csv'


class SplitType(Enum):
    PROBABILISTIC = 1
    LOO = 2
    LOO_CYTHON = 3


def set_seed(seed):
    print('seed = {0}'.format(seed))
    os.environ['RECSYS_SEED'] = str(seed)
    np.random.seed(seed)


def get_seed():
    env = os.getenv('RECSYS_SEED')
    if env:
        return int(env)
    return -1


def build_urm():
    urm_data = load_csv(DataFiles.TRAIN)
    urm_data = [[int(row[i]) if i <= 1 else int(float(row[i])) for i in range(len(row))] for row in urm_data]
    users, items, ratings = map(np.array, zip(*urm_data))
    return sps.csr_matrix((ratings, (users, items)))


def clusterize():
    data = load_csv(DataFiles.CLUSTERS)
    data = [[int(row[i]) for i in range(len(row))] for row in data]
    _, user_ids, cluster_ids = map(list, zip(*data))
    assert len(user_ids) == len(cluster_ids)
    data_len = len(user_ids)
    clusters = dict()
    for n in range(max(cluster_ids) + 1):
        clusters[n] = list()
    for i in range(data_len):
        user_id = user_ids[i]
        cluster_id = cluster_ids[i]
        clusters[cluster_id].append(user_id)
    return clusters


def get_cold_users(urm_train, return_warm=False):
    profile_lengths = np.ediff1d(urm_train.indptr)
    cold_users = np.where(profile_lengths == 0)[0]
    if return_warm:
        warm_users = np.where(profile_lengths > 0)[0]
        return cold_users, warm_users
    return cold_users


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


def build_age_ucm(n_users):
    age_ucm_users, age_ucm_features, age_ucm_values = __load_icm_csv(DataFiles.UCM_AGE, third_type=float)
    n_features = max(age_ucm_features) + 1
    shape = (n_users, n_features)
    age_ucm = sps.csr_matrix((age_ucm_values, (age_ucm_users, age_ucm_features)), shape=shape, dtype=int)
    return age_ucm


def build_region_ucm(n_users):
    region_ucm_users, region_ucm_features, region_ucm_values = __load_icm_csv(DataFiles.UCM_REGION, third_type=float)
    n_features = max(region_ucm_features) + 1
    shape = (n_users, n_features)
    region_ucm = sps.csr_matrix((region_ucm_values, (region_ucm_users, region_ucm_features)), shape=shape, dtype=int)
    return region_ucm


def build_ucm(n_users):
    age_ucm = build_age_ucm(n_users)
    region_ucm = build_region_ucm(n_users)
    return sps.hstack((age_ucm, region_ucm))


def build_target_users():
    target_users = load_csv(DataFiles.TARGET_USERS_TEST)
    return [int(x[0]) for x in target_users]


def build_all_matrices():
    urm = build_urm()
    n_users, n_items = urm.shape
    icm = build_icm(n_items)
    ucm = build_ucm(n_users)
    target_users = build_target_users()
    return urm, icm, ucm, target_users


def train_test_split(urm, split_type=SplitType.PROBABILISTIC, split=0.8):
    if split_type == SplitType.PROBABILISTIC:
        return __train_test_split(urm, split)
    elif split_type == SplitType.LOO:
        return __train_test_loo_split(urm)
    elif split_type == SplitType.LOO_CYTHON:
        return __train_test_loo_split_cython(urm)


def evaluate(recommender, urm_test, excluded_users=[], cython=False, verbose=True):
    from evaluation import evaluate_algorithm
    if cython:
        if verbose:
            print('Ignoring argument excluded_users')
        from cython_modules.evaluation import evaluate_cython
        if verbose:
            print('Using Cython evaluation')
        return evaluate_cython(recommender, urm_test, verbose=verbose)
    else:
        return evaluate_algorithm(recommender, urm_test, excluded_users=excluded_users, verbose=verbose)


def evaluate_mp(recommender, urm_tests, excluded_users=[], cython=False, verbose=True, n_processes=0):
    assert type(urm_tests) == list
    assert len(urm_tests) >= 1
    assert type(n_processes) == int
    if n_processes == 0:
        n_processes = len(urm_tests)
    with Pool(processes=n_processes) as pool:
        args = [(recommender, urm_test, excluded_users, cython, verbose) for urm_test in urm_tests]
        maps = pool.starmap(evaluate, args, chunksize=1)
        maps = [x['MAP'] for x in maps]
        return np.mean(maps)


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


group_struct = namedtuple('group_struct', ['in_group', 'not_in_group'])


def user_segmenter(urm_train, n_groups=10):
    groups = dict()
    users = dict()
    profile_length = np.ediff1d(urm_train.indptr)
    group_size = int(profile_length.size/n_groups)
    sorted_users = np.argsort(profile_length)
    for group_id in range(n_groups):
        start_pos = group_id * group_size
        end_pos = min((group_id + 1) * group_size, len(profile_length))
        users_in_group = sorted_users[start_pos:end_pos]
        for user in users_in_group:
            users[user] = group_id
        users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
        users_not_in_group = sorted_users[users_not_in_group_flag]
        groups[group_id] = group_struct(in_group=users_in_group, not_in_group=users_not_in_group)
    return groups, users


def multiple_splitting(seeds=(4951, 893, 2618, 39, 4947)):
    urm, icm, ucm, target_users = build_all_matrices()
    trains = list()
    tests = list()
    for seed in seeds:
        set_seed(seed)
        urm_train, urm_test = train_test_split(urm)
        trains.append(urm_train)
        tests.append(urm_test)
    return trains, tests, seeds


if __name__ == '__main__':
    from evaluation import evaluate_by_cluster
    from cf import ItemCFKNNRecommender
    from basic_recommenders import TopPopRecommender

    np.random.seed(42)
    urm, icm, ucm, target_users = build_all_matrices()
    urm_train, urm_test = train_test_split(urm, SplitType.PROBABILISTIC)
    top_pop = TopPopRecommender()
    top_pop.fit(urm_train)
    cf = ItemCFKNNRecommender(fallback_recommender=top_pop)
    cf.fit(urm_train, top_k=690, shrink=66, normalize=False, similarity='tanimoto')
    evaluate_by_cluster(cf, urm_test, clusterise())