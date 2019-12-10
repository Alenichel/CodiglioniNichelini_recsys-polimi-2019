#!/usr/bin/env python3

import numpy as np
import scipy.sparse as sps
from tqdm import trange
from run_utils import build_all_matrices, train_test_split, SplitType
from basic_recommenders import TopPopRecommender
from pprint import pprint as pp


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


def evaluate_algorithm(recommender_object, urm_test, at=10, excluded_users=[]):
    cumulative_precision = 0.0                              
    cumulative_recall = 0.0
    cumulative_MAP = 0.0
    num_eval = 0
    urm_test = urm_test.tocsr()
    n_users = urm_test.shape[0]
    for user_id in trange(n_users, desc='Evaluation'):
        if user_id not in excluded_users:
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


def multiple_evaluation(rec_sys, parameters, round=5, interesting_threshold=np.inf, use_group_evaluation=False):
    from cbf import ItemCBFKNNRecommender
    urm, icm, target_users = build_all_matrices()
    cumulativeMAP = 0
    report = {
        'max': 0,
        'min': np.inf,
        'median_value': 0,
        'param': str(parameters)
    }
    for x in range(round):
        urm_train, urm_test = train_test_split(urm, SplitType.LOO)
        n_users, n_items = urm_train.shape
        try:                                                        # check for fallback rc
            if rec_sys.fallback_recommender:
                rec_sys.fallback_recommender.fit(urm_train)
        except AttributeError:                                      # some rcs don't have a fallback rc
            pass

        if isinstance(rec_sys, ItemCBFKNNRecommender):              # custom path for cbf (it needs icm)
            rec_sys.fit(urm_train, icm, *parameters)
        else:
            rec_sys.fit(urm_train, *parameters)

        if use_group_evaluation:
            roundMAP = select_group_for_evaluation(rec_sys, urm_train, urm_test)['MAP']
        else:
            roundMAP = evaluate_algorithm(rec_sys, urm_test)['MAP']

        if roundMAP > report['max']:
            report['max'] = roundMAP
        elif roundMAP < report['min']:
            report['min'] = roundMAP
        cumulativeMAP += roundMAP
        report['median_value'] = cumulativeMAP / (x + 1)

        if report['median_value'] < interesting_threshold:      # if the median is too low, it's not useful to try again
            print("Exiting after: %d attempt. The value is not interesting" % (x+1))
            break

        print("median value so far %f (ROUND %d)" % (report['median_value'], x + 1))
    pp(report)
    return(report)


def select_group_for_evaluation(recommender_object, urm_train, urm_test, at=10):
    from user_hybrid import UserSegmenter
    in_groups, not_in_groups = UserSegmenter.real_segmenter(urm_train)
    returning = evaluate_algorithm(recommender_object, urm_test, excluded_users=not_in_groups[9])
    return returning
