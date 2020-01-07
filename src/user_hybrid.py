#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from run_utils import set_seed, build_all_matrices, train_test_split, SplitType, evaluate
from implicit_mf import LMFRecommender
from hybrid import get_hybrid_fallback


class UserSegmenter:

    def __init__(self, recommenders, urm_train, urm_test):
        self.recommenders = recommenders
        self.urm_train = urm_train
        self.urm_test = urm_test

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
    set_seed(42)
    EXPORT = False
    urm, icm, ucm, target_users = build_all_matrices()
    if EXPORT:
        urm_train = urm.tocsr()
        urm_test = None
    else:
        urm_train, urm_test = train_test_split(urm, SplitType.LOO_CYTHON)

    hybrid_fb = get_hybrid_fallback(urm_train, ucm)
    lmf1 = LMFRecommender()
    lmf1.fit(urm_train, alpha=7.497858, factors=115, regularization=9.942957)
    lmf2 = LMFRecommender()
    lmf2.fit(urm_train, alpha=6.314, factors=127, iterations=95, regularization=8.288)

    recommenders = [hybrid_fb, lmf1, lmf2]

    user_segmenter = UserSegmenter(recommenders, urm_train, urm_test)
    user_segmenter.analyze(group_size_percent=0.05)
