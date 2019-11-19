#!/usr/bin/env python3

from argparse import ArgumentParser
import numpy as np
import scipy.sparse as sps
from evaluation import evaluate_algorithm
from load_export_csv import load_csv, export_csv
from basic_recommenders import RandomRecommender, TopPopRecommender
from collaborative_filtering import ItemCFKNNRecommender


class Runner:

    def __init__(self):
        self.urm = None
        self.target_users = None

    def prepare_data(self, datafile='data/data_train.csv', datatargetfile='data/data_target_users_test.csv', evaluate=False):
        data = load_csv(datafile)
        data = [[int(row[i]) if i <= 1 else int(float(row[i])) for i in range(len(row))] for row in data]
        self.target_users = load_csv(datatargetfile)
        self.target_users = [int(x[0]) for x in self.target_users]
        users, items, ratings = map(np.array, zip(*data))
        self.urm = sps.coo_matrix((ratings, (users, items)))
        if evaluate:
            return self.train_test_split()
        return self.urm, None

    def train_test_split(self, split=0.8):
        print('Using stochastic splitting ({0:.2f}/{1:.2f})'.format(split, 1-split))
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
        return urm_train, urm_test

    def run(self, recommender, evaluate=False, export=False):
        print('Preparing data...')
        urm_train, urm_test = self.prepare_data(evaluate=evaluate)
        print('OK\nFitting...')
        recommender.fit(urm_train)
        print('OK')
        if evaluate:
            evaluate_algorithm(urm_test, recommender)
        if export:
            print('Exporting recommendations...', end='')
            data = [(u_id, recommender.recommend(u_id, at=10)) for u_id in self.target_users]
            export_csv(('user_id', 'item_list'), data)
            print('OK')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('recommender', choices=['random', 'toppop', 'cf'])
    parser.add_argument('--evaluate', '-e', action='store_true')
    parser.add_argument('--no-export', action='store_false')
    args = parser.parse_args()
    recommender = None
    if args.recommender == 'random':
        print('Using Random')
        recommender = RandomRecommender()
    elif args.recommender == 'toppop':
        print('Using TopPop')
        recommender = TopPopRecommender()
    elif args.recommender == 'cf':
        print('Using Collaborative Filtering (item-based)')
        recommender = ItemCFKNNRecommender()
    Runner().run(recommender, args.evaluate, args.no_export)
