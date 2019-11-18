#!/usr/bin/env python3

from argparse import ArgumentParser
import numpy as np
import scipy.sparse as sps
from evaluation import evaluate_algorithm
from load_export_csv import load_csv
from random_recommender import RandomRecommender


class Runner:

    def __init__(self):
        self.users = None
        self.items = None
        self.ratings = None

    def prepare_data(self):
        data = load_csv('data/data_train.csv')
        data = [[int(row[i]) if i <= 1 else float(row[i]) for i in range(len(row))] for row in data]
        self.users, self.items, self.ratings = map(np.array, zip(*data))
        return self.train_test_split()

    def train_test_split(self, split=0.8):
        train_mask = np.random.choice([True, False], self.users.size, p=[split, 1-split])
        urm_train = sps.csr_matrix((self.ratings[train_mask], (self.users[train_mask], self.items[train_mask])))
        test_mask = np.logical_not(train_mask)
        urm_test = sps.csr_matrix((self.ratings[test_mask], (self.users[test_mask], self.items[test_mask])))
        return urm_train, urm_test

    def run(self, recommender):
        urm_train, urm_test = self.prepare_data()
        recommender.fit(urm_train)
        evaluate_algorithm(urm_test, recommender)
        # TODO: Export data


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('recommender', choices=['random'])
    args = parser.parse_args()
    if args.recommender == 'random':
        recommender = RandomRecommender()
    Runner().run(recommender)
