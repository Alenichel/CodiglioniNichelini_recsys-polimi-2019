#!/usr/bin/env python3

import numpy as np
from basic_recommenders import TopPopRecommender
from cbf import UserCBFKNNRecommender
from hybrid import MergingTechniques, HybridRecommender
from implicit_mf import LMFRecommender
from run_utils import set_seed, build_all_matrices, train_test_split, evaluate


if __name__ == '__main__':
    set_seed(42)
    urm, icm, ucm, target_users = build_all_matrices()
    urm_train, urm_test = train_test_split(urm)

    # TOP-POP
    top_pop = TopPopRecommender()
    top_pop.fit(urm_train)
    # USER CBF
    user_cbf = UserCBFKNNRecommender()
    user_cbf.fit(urm_train, ucm, top_k=496, shrink=0, normalize=False)
    # LMF
    lmf1 = LMFRecommender()
    lmf1.fit(urm_train, alpha=7.497858, factors=115, regularization=9.942957)
    lmf2 = LMFRecommender()
    lmf2.fit(urm_train, alpha=6.314, factors=127, iterations=95, regularization=8.288)

    profile_lengths = np.ediff1d(urm_train.indptr)
    warm_users = np.where(profile_lengths != 0)[0]

    results = dict()

    for n in range(128):
        b = '{0:07b}'.format(n)
        merging = b[4:]
        if merging == '001':
            merging = MergingTechniques.FREQ
        elif merging == '010':
            merging = MergingTechniques.RR
        elif merging == '100':
            merging = MergingTechniques.MEDRANK
        else:
            continue
        recommenders = []
        if b[0] == '1':
            recommenders.append(top_pop)
        if b[1] == '1':
            recommenders.append(user_cbf)
        if b[2] == '1':
            recommenders.append(lmf1)
        if b[3] == '1':
            recommenders.append(lmf2)
        if len(recommenders) == 0:
            continue
        print(b)
        hybrid_fb = HybridRecommender(recommenders, urm_train, merging_type=merging)
        results[b] = evaluate(hybrid_fb, urm_test, excluded_users=warm_users)['MAP']

    sorted_keys = sorted(results, key=lambda k: results[k])
    for k in sorted_keys:
        print(k, results[k])
