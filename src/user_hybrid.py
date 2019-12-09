#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from run_utils import build_all_matrices, train_test_split, SplitType, export, evaluate
from cf import ItemCFKNNRecommender, UserCFKNNRecommender
from cbf import ItemCBFKNNRecommender
from basic_recommenders import TopPopRecommender


if __name__ == '__main__':
    EXPORT = False
    urm, icm, target_users = build_all_matrices()
    if EXPORT:
        urm_train = urm.tocsr()
        urm_test = None
    else:
        urm_train, urm_test = train_test_split(urm, SplitType.LOO_CYTHON)

    top_pop = TopPopRecommender()
    top_pop.fit(urm_train)
    item_cbf = ItemCBFKNNRecommender()
    item_cbf.fit(urm_train, icm)
    user_cf = UserCFKNNRecommender()
    user_cf.fit(urm_train)
    item_cf = ItemCFKNNRecommender()
    item_cf.fit(urm_train)

    profile_length = np.ediff1d(urm_train.indptr)
    group_size_percent = 0.5
    n_groups = int(1 / group_size_percent)
    group_size = int(profile_length.size * group_size_percent)
    sorted_users = np.argsort(profile_length)

    MAP_itemKNNCF_per_group = []
    MAP_itemKNNCBF_per_group = []
    MAP_userKNNCF_per_group = []
    MAP_topPop_per_group = []

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

        results = evaluate(item_cf, urm_test, excluded_users=users_not_in_group)
        MAP_itemKNNCF_per_group.append(results["MAP"])

        results = evaluate(user_cf, urm_test, excluded_users=users_not_in_group)
        MAP_userKNNCF_per_group.append(results["MAP"])

        results = evaluate(item_cbf, urm_test, excluded_users=users_not_in_group)
        MAP_itemKNNCBF_per_group.append(results["MAP"])

        results = evaluate(top_pop, urm_test, excluded_users=users_not_in_group)
        MAP_topPop_per_group.append(results["MAP"])

    for group_id in range(n_groups):
        start_pos = group_id * group_size
        end_pos = min((group_id + 1) * group_size, len(profile_length))
        users_in_group = sorted_users[start_pos:end_pos]
        users_in_group_p_len = profile_length[users_in_group]
        print('Group {}, average p.len {:.2f}, min {}, max {}'.format(group_id,
                                                                      users_in_group_p_len.mean(),
                                                                      users_in_group_p_len.min(),
                                                                      users_in_group_p_len.max()))

    plt.plot(MAP_itemKNNCF_per_group, label="Item CF")
    plt.plot(MAP_itemKNNCBF_per_group, label="Item CBF")
    plt.plot(MAP_userKNNCF_per_group, label="User CF")
    plt.plot(MAP_topPop_per_group, label="TopPop")
    plt.ylabel('MAP')
    plt.xlabel('User Group')
    plt.xticks(np.arange(n_groups))
    plt.legend()
    plt.show()
