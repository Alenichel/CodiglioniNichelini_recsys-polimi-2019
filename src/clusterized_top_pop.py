#!/usr/bin/env python3

import numpy as np
from tqdm import trange
from run_utils import build_all_matrices, clusterize, train_test_split, SplitType, export, evaluate
from basic_recommenders import TopPopRecommender
from cbf import UserCBFKNNRecommender


class ClusterizedTopPop:

    def __init__(self):
        self.urm_train = None
        self.ucm = None
        self.recommenders = dict()
        self.cluster_for_user = dict()
        self.std_top_pop = None

    def fit(self, urm_train):
        self.urm_train = urm_train
        clusters = clusterize()
        for cluster in clusters:
            users = clusters[cluster]
            for user_id in users:
                self.cluster_for_user[user_id] = cluster
            filtered_urm = self.urm_train[users, :]
            top_pop = TopPopRecommender()
            top_pop.fit(filtered_urm)
            self.recommenders[cluster] = top_pop

    def recommend(self, user_id, at=None, exclude_seen=True):
        cluster = self.cluster_for_user[user_id]
        return self.recommenders[cluster].recommend(user_id, at, False)


if __name__ == '__main__':
    np.random.seed(42)
    EXPORT = False
    urm, icm, ucm, target_users = build_all_matrices()
    if EXPORT:
        urm_train = urm.tocsr()
        urm_test = None
    else:
        urm_train, urm_test = train_test_split(urm, SplitType.PROBABILISTIC)
    # TOP-POP
    top_pop = ClusterizedTopPop()
    top_pop.fit(urm_train)
    # USER CBF
    user_cbf = UserCBFKNNRecommender()
    user_cbf.fit(urm_train, ucm, top_k=496, shrink=0, normalize=False)
    # HYBRID FALLBACK
    from hybrid import HybridRecommender, MergingTechniques
    hybrid_fb = HybridRecommender([top_pop, user_cbf], urm_train, merging_type=MergingTechniques.MEDRANK)
    if EXPORT:
        export(target_users, hybrid_fb)
    else:
        profile_lengths = np.ediff1d(urm_train.indptr)
        warm_users = np.where(profile_lengths != 0)[0]
        evaluate(hybrid_fb, urm_test, excluded_users=warm_users)
