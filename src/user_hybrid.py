#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from run_utils import build_all_matrices, train_test_split, SplitType, evaluate
from basic_recommenders import TopPopRecommender, RandomRecommender
from cbf import ItemCBFKNNRecommender
from cf import ItemCFKNNRecommender, UserCFKNNRecommender
from slim_bpr import SLIM_BPR


if __name__ == '__main__':
    EXPORT = False
    urm, icm, target_users = build_all_matrices()
    if EXPORT:
        urm_train = urm.tocsr()
        urm_test = None
    else:
        urm_train, urm_test = train_test_split(urm, SplitType.LOO_CYTHON)

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

    profile_length = np.ediff1d(urm_train.indptr)
    group_size_percent = 0.05
    n_groups = int(1 / group_size_percent)
    group_size = int(profile_length.size * group_size_percent)
    sorted_users = np.argsort(profile_length)

    random_map = []
    top_pop_map = []
    item_cbf_map = []
    user_cf_map = []
    item_cf_map = []
    slim_bpr_map = []

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

        results = evaluate(random, urm_test, excluded_users=users_not_in_group)
        random_map.append(results['MAP'])

        results = evaluate(top_pop, urm_test, excluded_users=users_not_in_group)
        top_pop_map.append(results['MAP'])

        results = evaluate(item_cbf, urm_test, excluded_users=users_not_in_group)
        item_cbf_map.append(results['MAP'])

        results = evaluate(user_cf, urm_test, excluded_users=users_not_in_group)
        user_cf_map.append(results['MAP'])

        results = evaluate(item_cf, urm_test, excluded_users=users_not_in_group)
        item_cf_map.append(results['MAP'])

        results = evaluate(slim_bpr, urm_test, excluded_users=users_not_in_group)
        slim_bpr_map.append(results['MAP'])

    for group_id in range(n_groups):
        start_pos = group_id * group_size
        end_pos = min((group_id + 1) * group_size, len(profile_length))
        users_in_group = sorted_users[start_pos:end_pos]
        users_in_group_p_len = profile_length[users_in_group]
        print('Group {}, average p.len {:.2f}, min {}, max {}'.format(group_id,
                                                                      users_in_group_p_len.mean(),
                                                                      users_in_group_p_len.min(),
                                                                      users_in_group_p_len.max()))

    plt.plot(random_map, label='Random')
    plt.plot(top_pop_map, label='TopPop')
    plt.plot(item_cbf_map, label='Item CBF')
    plt.plot(user_cf_map, label='User CF')
    plt.plot(item_cf_map, label='Item CF')
    plt.plot(slim_bpr_map, label='SLIM BPR')
    plt.ylabel('MAP')
    plt.xlabel('User Group')
    plt.xticks(np.arange(n_groups))
    plt.legend()
    plt.show()
