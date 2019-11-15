#!/usr/bin/env python3

import numpy as np
import scipy.sparse as sps


def train_test_split(users, items, ratings, nnz, split=0.8):
    train_mask = np.random.choice([True, False], nnz, p=[split, 1-split])             # randomly create the mask
    urm_train = sps.csr_matrix((ratings[train_mask], (users[train_mask], items[train_mask])))      # we get the training set
    test_mask = np.logical_not(train_mask)
    urm_test = sps.csr_matrix((ratings[test_mask], (users[test_mask], items[test_mask])))          # we get the test set
    return urm_train, urm_test