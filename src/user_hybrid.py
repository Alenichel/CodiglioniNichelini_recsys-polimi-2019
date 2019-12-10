#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from run_utils import build_all_matrices, train_test_split, SplitType, evaluate
from basic_recommenders import TopPopRecommender, RandomRecommender
from cbf import ItemCBFKNNRecommender
from cf import ItemCFKNNRecommender, UserCFKNNRecommender
from slim_bpr import SLIM_BPR


class UserSegmenter:

    def __init__(self, recommenders, urm_train, urm_test):
        self.recommenders = recommenders
        self.urm_train = urm_train
        self.urm_test = urm_test

    @staticmethod
    def real_segmenter(urm_train, group_size_percent=0.1):
        profile_length = np.ediff1d(urm_train.indptr)
        n_groups = int(1 / group_size_percent)
        group_size = int(profile_length.size * group_size_percent)
        sorted_users = np.argsort(profile_length)
        users_in_groups = []
        users_not_in_groups = []
        for group_id in range(n_groups):
            start_pos = group_id * group_size
            end_pos = min((group_id + 1) * group_size, len(profile_length))
            users_in_group = sorted_users[start_pos:end_pos]
            users_in_groups += [users_in_group]
            users_in_group_p_len = profile_length[users_in_group]
            print('Group {}, average p.len {:.2f}, min {}, max {}'.format(group_id,
                                                                          users_in_group_p_len.mean(),
                                                                          users_in_group_p_len.min(),
                                                                          users_in_group_p_len.max()))
            users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
            users_not_in_group = sorted_users[users_not_in_group_flag]
            users_not_in_groups += [users_not_in_group]
        return users_in_groups, users_not_in_groups

    def analyze(self, group_size_percent=0.1):
        profile_length = np.ediff1d(self.urm_train.indptr)
        n_groups = int(1 / group_size_percent)
        group_size = int(profile_length.size * group_size_percent)
        sorted_users = np.argsort(profile_length)
        maps = [[] for r in self.recommenders]
        for group_id in range(n_groups):
            start_pos = group_id * group_size
            end_pos = min((group_id + 1) * group_size, len(profile_length))
            users_in_group = sorted_users[start_pos:end_pos]
            users_in_group_p_len = profile_length[users_in_group]
            print('Group {}, average p.len {:.2f}, min {}, max {}'.format(group_id,
                                                                          users_in_group_p_len.mean(),
                                                                          users_in_group_p_len.min(),
                                                                          users_in_group_p_len.max()))
            users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
            users_not_in_group = sorted_users[users_not_in_group_flag]
            for idx, rec in enumerate(self.recommenders):
                map_value = evaluate(rec, self.urm_test, excluded_users=users_not_in_group)['MAP']
                maps[idx].append(map_value)
        for group_id in range(n_groups):
            start_pos = group_id * group_size
            end_pos = min((group_id + 1) * group_size, len(profile_length))
            users_in_group = sorted_users[start_pos:end_pos]
            users_in_group_p_len = profile_length[users_in_group]
            print('Group {}, average p.len {:.2f}, min {}, max {}'.format(group_id,
                                                                          users_in_group_p_len.mean(),
                                                                          users_in_group_p_len.min(),
                                                                          users_in_group_p_len.max()))
        for idx, rec in enumerate(self.recommenders):
            plt.plot(maps[idx], label=str(rec))
        plt.ylabel('MAP')
        plt.xlabel('User Group')
        plt.xticks(np.arange(n_groups))
        plt.legend()
        plt.show()


if __name__ == '__main__':
    EXPORT = False
    urm, icm, target_users = build_all_matrices()
    if EXPORT:
        urm_train = urm.tocsr()
        urm_test = None
    else:
        urm_train, urm_test = train_test_split(urm, SplitType.LOO_CYTHON)

    '''
    random = RandomRecommender()
    random.fit(urm_train)
    top_pop = TopPopRecommender()
    top_pop.fit(urm_train)
    item_cbf = ItemCBFKNNRecommender()
    item_cbf.fit(urm_train, icm)
    user_cf = UserCFKNNRecommender()
    user_cf.fit(urm_train)
    item_cf = ItemCFKNNRecommender()
    item_cf.fit(urm_train, top_k=5, shrink=20, similarity='tanimoto')
    slim_bpr = SLIM_BPR()
    slim_bpr.fit(urm_train, epochs=100)
    recommenders = [random, top_pop, item_cbf, user_cf, item_cf, slim_bpr]
    '''

    rec1 = UserCFKNNRecommender()
    rec1.fit(urm_train, top_k=551, shrink=702, similarity='cosine')
    rec2 = UserCFKNNRecommender()
    rec2.fit(urm_train, top_k=50, shrink=100, similarity='jaccard')
    rec3 = ItemCFKNNRecommender()
    rec3.fit(urm_train, top_k=5, shrink=20, similarity='tanimoto')
    recommenders = [rec1, rec2, rec3]

    user_segmenter = UserSegmenter(recommenders, urm_train, urm_test)
    user_segmenter.analyze(group_size_percent=0.05)
