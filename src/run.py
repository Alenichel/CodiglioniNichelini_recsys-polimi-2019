#!/usr/bin/env python3

from argparse import ArgumentParser
import numpy as np
import scipy.sparse as sps
from evaluation import evaluate_algorithm
from load_export_csv import load_csv, export_csv
from basic_recommenders import RandomRecommender, TopPopRecommender
from cbf import ItemCBFKNNRecommender
from cf import ItemCFKNNRecommender
from slim import SLIM_BPR_Recommender
from slim_new import SLIM_BPR


class DataFiles:
    TRAIN = 'data/data_train.csv'
    TARGET_USERS_TEST = 'data/data_target_users_test.csv'
    ICM_ASSET = 'data/data_ICM_asset.csv'
    ICM_PRICE = 'data/data_ICM_price.csv'
    ICM_SUBCLASS = 'data/data_ICM_sub_class.csv'
    UCM_AGE = 'data/data_UCM_age.csv'
    UCM_REGION = 'data/data_UCM_region.csv'


class Runner:

    def __init__(self, recommender, evaluate=True, split='prob', export=False):
        self.urm = None
        self.icm = None
        self.target_users = None
        self.recommender = recommender
        self.evaluate = evaluate
        self.split = split
        self.export = export

    @staticmethod
    def load_icm_csv(filename):
        data = load_csv(filename)
        data = [[int(row[i]) if i <= 1 else float(row[i]) for i in range(len(row))] for row in data]
        items, features, values = map(np.array, zip(*data))
        return items, features, values

    def prepare_data(self):
        urm_data = load_csv(DataFiles.TRAIN)
        urm_data = [[int(row[i]) if i <= 1 else int(float(row[i])) for i in range(len(row))] for row in urm_data]
        users, items, ratings = map(np.array, zip(*urm_data))
        self.urm = sps.coo_matrix((ratings, (users, items)))
        price_icm_items, price_icm_features, price_icm_values = Runner.load_icm_csv(DataFiles.ICM_PRICE)
        asset_icm_items, asset_icm_features, asset_icm_values = Runner.load_icm_csv(DataFiles.ICM_ASSET)
        asset_icm_features += 1     # asset had feature number 0, now it's 1
        subclass_icm_items, subclass_icm_features, subclass_icm_values = Runner.load_icm_csv(DataFiles.ICM_SUBCLASS)
        subclass_icm_features += 1  # subclass numbers starts from 1, so we add 1 to make room for price and asset
        icm_items = np.concatenate((price_icm_items, asset_icm_items, subclass_icm_items))
        icm_features = np.concatenate((price_icm_features, asset_icm_features, subclass_icm_features))
        icm_values = np.concatenate((price_icm_values, asset_icm_values, subclass_icm_values))
        self.icm = sps.csr_matrix((icm_values, (icm_items, icm_features)))
        self.target_users = load_csv(DataFiles.TARGET_USERS_TEST)
        self.target_users = [int(x[0]) for x in self.target_users]
        if self.evaluate:
            if self.split == 'prob':
                return self.train_test_split()
            elif self.split == 'loo':
                return self.train_test_loo_split()
        return self.urm, None

    def train_test_split(self, split=0.8):
        print('Using probabilistic splitting ({0:.2f}/{1:.2f})'.format(split, 1-split))
        num_interactions = self.urm.nnz
        shape = self.urm.shape
        train_mask = np.random.choice([True, False], num_interactions, p=[split, 1-split])
        urm_train = sps.coo_matrix((self.urm.data[train_mask], (self.urm.row[train_mask], self.urm.col[train_mask])), shape=shape)
        urm_train = urm_train.tocsr()
        test_mask = np.logical_not(train_mask)
        urm_test = sps.coo_matrix((self.urm.data[test_mask], (self.urm.row[test_mask], self.urm.col[test_mask])), shape=shape)
        urm_test = urm_test.tocsr()
        return urm_train, urm_test

    def train_test_loo_split(self):
        print('Using LeaveOneOut')
        urm = self.urm.tocsr()
        users_len = len(urm.indptr) - 1
        items_len = max(urm.indices) + 1
        urm_train = urm.copy()
        urm_test = np.zeros((users_len, items_len))
        for user_id in range(users_len):
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
        print('OK\nFitting...')
        if requires_icm:
            recommender.fit(urm_train, self.icm)
        else:
            recommender.fit(urm_train)
        print('OK')
        if self.evaluate:
            evaluate_algorithm(urm_test, recommender)
        if self.export:
            print('Exporting recommendations...', end='')
            data = [(u_id, recommender.recommend(u_id, at=10)) for u_id in self.target_users]
            export_csv(('user_id', 'item_list'), data)
            print('OK')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('recommender', choices=['random', 'top-pop', 'cbf', 'cf', 'slim-bpr'])
    parser.add_argument('--evaluate', '-e', action='store_true')
    parser.add_argument('--split', '-s', choices=['prob', 'loo'], default='prob')
    parser.add_argument('--no-export', action='store_false')
    args = parser.parse_args()
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
    elif args.recommender == 'slim-bpr':
        print('Using SLIM (BPR)')
        recommender = SLIM_BPR()
    Runner(recommender, args.evaluate, args.split, args.no_export).run(requires_icm=(args.recommender == 'cbf'))
