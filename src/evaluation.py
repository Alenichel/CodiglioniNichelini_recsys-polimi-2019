#!/usr/bin/env python3

import numpy as np
import scipy.sparse as sps


def precision(is_relevant, relevant_items):
    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)
    return precision_score


def recall(is_relevant, relevant_items):
    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]
    return recall_score


def MAP(is_relevant, relevant_items):
    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])
    return map_score


def evaluate_algorithm(urm_test, recommender_object, at=10):
    cumulative_precision = 0.0                              
    cumulative_recall = 0.0
    cumulative_MAP = 0.0
    num_eval = 0
    urm_test = sps.csr_matrix(urm_test)
    n_users = urm_test.shape[0]
    for user_id in range(n_users):
        if user_id % 5000 == 0:
            print("Evaluated user {} of {}".format(user_id, n_users))
        start_pos = urm_test.indptr[user_id]
        end_pos = urm_test.indptr[user_id+1]
        if end_pos - start_pos > 0:
            relevant_items = urm_test.indices[start_pos:end_pos]
            recommended_items = recommender_object.recommend(user_id, at=at)
            num_eval += 1
            is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
            cumulative_precision += precision(is_relevant, relevant_items)
            cumulative_recall += recall(is_relevant, relevant_items)
            cumulative_MAP += MAP(is_relevant, relevant_items)
    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_MAP /= num_eval
    print('Recommender performance is:')
    print('    Precision = {:.5f}'.format(cumulative_precision))
    print('    Recall    = {:.5f}'.format(cumulative_recall))
    print('    MAP       = {:.5f}'.format(cumulative_MAP))
    result_dict = {
        "precision": cumulative_precision,                          
        "recall": cumulative_recall,
        "MAP": cumulative_MAP,
    }
    return result_dict
