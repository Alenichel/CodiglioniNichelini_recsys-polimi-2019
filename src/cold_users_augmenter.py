#!/usr/bin/env python3

import numpy as np
from tqdm import tqdm, trange
from cbf import UserCBFKNNRecommender
from hybrid import get_hybrid
from run_utils import set_seed, build_all_matrices, train_test_split, evaluate, get_cold_users


class ColdUsersAugmenter:

    def __init__(self, urm_train, ucm, top_k=5, shrink=0, normalize=False, similarity='dice', verbose=True):
        self.urm_train = urm_train.copy().tolil()
        cbf = UserCBFKNNRecommender()
        cbf.fit(urm_train, ucm, top_k=top_k*10, shrink=shrink, normalize=normalize, similarity=similarity)
        self.S = cbf.w_sparse.T.tocsr()
        self.cold_users, self.warm_users = get_cold_users(urm_train, return_warm=True)
        similar_users = list()
        n_users = urm_train.shape[0]
        for user_id in trange(n_users, desc='Extracting similar users', disable=not verbose):
            users = np.array(self.S[user_id].todense()).squeeze()
            users = np.where(users != 0)[0]
            users = users[np.isin(users, self.cold_users, invert=True)]
            assert len(users) >= top_k
            users = users[:top_k]
            similar_users.append(users)
        self.similar_users = np.vstack(similar_users)
        for user_id in tqdm(self.cold_users, desc='Augmenting cold users', disable=not verbose):
            similar_users = self.similar_users[user_id]
            similar_interactions = urm_train[similar_users].sum(axis=0).astype(bool).astype(int)
            self.urm_train[user_id] += similar_interactions
        self.urm_train = self.urm_train.tocsr()


if __name__ == '__main__':
    set_seed(42)
    urm, icm, ucm, target_users = build_all_matrices()
    urm_train, urm_test = train_test_split(urm)

    augmenter = ColdUsersAugmenter(urm_train, ucm)
    urm_train = augmenter.urm_train

    hybrid = get_hybrid(urm_train, icm, ucm, fallback=False)

    print('Evaluating only on cold users')
    evaluate(hybrid, urm_test, excluded_users=augmenter.warm_users)

    print('Evaluating on all users')
    evaluate(hybrid, urm_test)
