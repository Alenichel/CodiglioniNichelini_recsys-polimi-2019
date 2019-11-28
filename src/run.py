#!/usr/bin/env python3

from argparse import ArgumentParser
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import LabelEncoder
from time import time
from datetime import timedelta
from evaluation import evaluate_algorithm
from load_export_csv import load_csv, export_csv
from basic_recommenders import RandomRecommender, TopPopRecommender
from cbf import ItemCBFKNNRecommender
from cf import ItemCFKNNRecommender
from slim_nb import SLIM_BPR_Recommender
from slim_bpr import SLIM_BPR


class DataFiles:
    TRAIN = 'data/data_train.csv'
    TARGET_USERS_TEST = 'data/data_target_users_test.csv'
    ICM_ASSET = 'data/data_ICM_asset.csv'
    ICM_PRICE = 'data/data_ICM_price.csv'
    ICM_SUBCLASS = 'data/data_ICM_sub_class.csv'
    UCM_AGE = 'data/data_UCM_age.csv'
    UCM_REGION = 'data/data_UCM_region.csv'


class Runner:

    def __init__(self, recommender, evaluate=True, split='loo', export=False, use_validation=False):
        self.urm = None
        self.icm = None
        self.target_users = None
        self.recommender = recommender
        self.evaluate = evaluate
        self.split = split
        self.export = export
        self.validation = use_validation

    @staticmethod
    def load_icm_csv(filename, third_type):
        data = load_csv(filename)
        data = [[int(row[i]) if i <= 1 else third_type(row[i]) for i in range(len(row))] for row in data]
        items, features, values = map(np.array, zip(*data))
        return items, features, values

    @staticmethod
    def encode_values(values):
        le = LabelEncoder()
        le.fit(values)
        return le.transform(values)

    def build_icm(self):
        # PRICE
        price_icm_items, _, price_icm_values = Runner.load_icm_csv(DataFiles.ICM_PRICE, third_type=float)
        price_icm_values = Runner.encode_values(price_icm_values)
        n_items = self.urm.shape[1]
        n_features = max(price_icm_values) + 1
        shape = (n_items, n_features)
        ones = np.ones(len(price_icm_values))
        price_icm = sps.csr_matrix((ones, (price_icm_items, price_icm_values)), shape=shape, dtype=int)

        # ASSET
        asset_icm_items, _, asset_icm_values = Runner.load_icm_csv(DataFiles.ICM_ASSET, third_type=float)
        asset_icm_values += 1
        asset_icm_values = Runner.encode_values(asset_icm_values)
        n_features = max(asset_icm_values) + 1
        shape = (n_items, n_features)
        ones = np.ones(len(asset_icm_values))
        asset_icm = sps.csr_matrix((ones, (asset_icm_items, asset_icm_values)), shape=shape, dtype=int)

        # SUBCLASS
        subclass_icm_items, subclass_icm_features, subclass_icm_values = Runner.load_icm_csv(DataFiles.ICM_SUBCLASS, third_type=float)
        n_features = max(subclass_icm_features) + 1
        shape = (n_items, n_features)
        subclass_icm = sps.csr_matrix((subclass_icm_values, (subclass_icm_items, subclass_icm_features)), shape=shape, dtype=int)

        self.icm = sps.hstack((price_icm, asset_icm, subclass_icm)).tocsr()

    def prepare_data(self):
        urm_data = load_csv(DataFiles.TRAIN)
        urm_data = [[int(row[i]) if i <= 1 else int(float(row[i])) for i in range(len(row))] for row in urm_data]
        users, items, ratings = map(np.array, zip(*urm_data))
        self.urm = sps.coo_matrix((ratings, (users, items)))
        self.build_icm()
        self.target_users = load_csv(DataFiles.TARGET_USERS_TEST)
        self.target_users = [int(x[0]) for x in self.target_users]
        if self.evaluate:
            if self.split == 'prob':
                return Runner.train_test_split(self.urm)
            elif self.split == 'loo':
                return self.train_test_loo_split(self.urm)
        return self.urm, None

    @staticmethod
    def train_test_split(urm, split=0.85):
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

    @staticmethod
    def train_test_loo_split(urm):
        print('Using LeaveOneOut')
        urm = urm.tocsr()
        num_users = urm.shape[0]
        num_items = urm.shape[1]
        urm_train = urm.copy()
        urm_test = np.zeros((num_users, num_items))
        for user_id in range(num_users):
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

    def run(self, requires_icm=False):
        print('Preparing data...')
        urm_train, urm_test = self.prepare_data()
        if self.validation:
            urm_train, urm_validation = Runner.train_test_split(urm_train)
        print('OK\nFitting...')
        if requires_icm:
            recommender.fit(urm_train, self.icm)
        elif self.validation:
            recommender.fit(urm_train, urm_validation)
        else:
            recommender.fit(urm_train)
        print('OK')
        if self.evaluate:
            print('Evaluating...')
            evaluate_algorithm(urm_test, recommender)
        if self.export:
            print('Exporting recommendations...')
            data = list()
            batch_size = 1000
            start_time = time()
            start_time_batch = start_time
            for u_id in self.target_users:
                data.append((u_id, recommender.recommend(u_id, at=10)))
                if u_id % batch_size == 0 and u_id > 0:
                    index_uid = self.target_users.index(u_id)
                    elapsed = timedelta(seconds=int(time() - start_time))
                    samples_ps = batch_size / (time() - start_time_batch)
                    eta = timedelta(seconds=int((len(self.target_users) - index_uid) / samples_ps))
                    print('Exported {0:7.0f} users ( {1:5.2f}% ) in {2} | Samples/s: {3:6.1f} | ETA: {4}'.format(
                        index_uid,
                        100.0 * float(index_uid) / len(self.target_users),
                        elapsed,
                        samples_ps,
                        eta))
                    start_time_batch = time()
            export_csv(('user_id', 'item_list'), data)
            print('OK')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('recommender', choices=['random', 'top-pop', 'cbf', 'cf', 'slim-nb', 'slim-bpr'])
    parser.add_argument('--evaluate', '-e', action='store_true')
    parser.add_argument('--split', '-s', choices=['prob', 'loo'], default='loo')
    parser.add_argument('--no-export', action='store_false')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--use-validation-set', action='store_true')
    args = parser.parse_args()
    if args.seed:
        print('Seeding random numbers generator with seed: {0}'.format(args.seed))
        np.random.seed(args.seed)
    recommender = None
    if args.recommender == 'random':
        print('Using Random')
        recommender = RandomRecommender()
    elif args.recommender == 'top-pop':
        print('Using TopPop')
        recommender = TopPopRecommender()
    elif args.recommender == 'cbf':
        print('Using Content-Based Filtering')
        recommender = ItemCBFKNNRecommender()
    elif args.recommender == 'cf':
        print('Using Collaborative Filtering (item-based)')
        recommender = ItemCFKNNRecommender()
    elif args.recommender == 'slim-nb':
        print('Using SLIM (from the notebook)')
        recommender = SLIM_BPR_Recommender()
    elif args.recommender == 'slim-bpr':
        print('Using SLIM (BPR)')
        recommender = SLIM_BPR()
    Runner(recommender, args.evaluate, args.split, args.no_export, args.use_validation_set).run(requires_icm=(args.recommender == 'cbf'))
