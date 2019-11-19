#!/usr/bin/env python3

from argparse import ArgumentParser
import numpy as np
import scipy.sparse as sps
from evaluation import evaluate_algorithm
from load_export_csv import load_csv, export_csv
from basic_recommenders import RandomRecommender, TopPopRecommender


class Runner:

    def __init__(self):
        self.users = None
        self.items = None
        self.ratings = None
        self.target_users = None

    def prepare_data(self, datafile='data/data_train.csv', datatargetfile='data/data_target_users_test.csv', split=False):
        data = load_csv(datafile)
        data = [[int(row[i]) if i <= 1 else float(row[i]) for i in range(len(row))] for row in data]
        self.target_users = load_csv(datatargetfile)
        self.target_users = [int(x[0]) for x in self.target_users]
        self.users, self.items, self.ratings = map(np.array, zip(*data))
        return self.train_test_split()

    def train_test_split(self, split=0.8):
        train_mask = np.random.choice([True, False], self.users.size, p=[split, 1-split])
        urm_train = sps.csr_matrix((self.ratings[train_mask], (self.users[train_mask], self.items[train_mask])))
        test_mask = np.logical_not(train_mask)
        urm_test = sps.csr_matrix((self.ratings[test_mask], (self.users[test_mask], self.items[test_mask])))
        return urm_train, urm_test

    def run(self, recommender, evaluate=False):
        urm_train, urm_test = self.prepare_data()
        recommender.fit(urm_train)
        if evaluate:
            evaluate_algorithm(urm_test, recommender)
        print('Exporting recommendations...', end='')
        data = [(u_id, recommender.recommend(u_id, at=10)) for u_id in self.target_users]
        export_csv(('user_id', 'item_list'), data)
        print('OK')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('recommender', choices=['random', 'toppop'])
    parser.add_argument('--evaluate', '-e', action='store_true')
    args = parser.parse_args()
    recommender = None
    if args.recommender == 'random':
        print('Using Random recommender')
        recommender = RandomRecommender()
    elif args.recommender == 'toppop':
        print('Using TopPop recommender')
        recommender = TopPopRecommender()
    Runner().run(recommender, args.evaluate)
