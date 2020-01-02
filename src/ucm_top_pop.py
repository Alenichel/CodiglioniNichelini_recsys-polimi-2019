#!/usr/bin/env python3

import numpy as np
from tqdm import trange
from run_utils import build_all_matrices, build_age_ucm, build_region_ucm, train_test_split, SplitType, export, evaluate
from basic_recommenders import TopPopRecommender
from cbf import UserCBFKNNRecommender


class TopPopUCM:

    def __init__(self):
        self.urm_train = None
        self.ucm = None
        self.recommenders = dict()
        self.feature_for_user = dict()
        self.std_top_pop = None

    def __users_for_feature(self, feature):
        users_mask = self.ucm[:, feature] == 1
        users_in_feature = users_mask.indices
        return users_in_feature

    def fit(self, urm_train, ucm):
        self.urm_train = urm_train
        self.ucm = ucm.tocsc()
        self.std_top_pop = TopPopRecommender()
        self.std_top_pop.fit(urm_train)
        for feature in trange(self.ucm.shape[1]):
            users = self.__users_for_feature(feature)
            for user_id in users:
                self.feature_for_user[user_id] = feature
            filtered_urm = self.urm_train[users, :]
            top_pop = TopPopRecommender()
            top_pop.fit(filtered_urm)
            self.recommenders[feature] = top_pop

    def recommend(self, user_id, at=None, exclude_seen=True):
        try:
            feature = self.feature_for_user[user_id]
            return self.recommenders[feature].recommend(user_id, at, exclude_seen)
        except (KeyError, IndexError):
            return self.std_top_pop.recommend(user_id, at, exclude_seen)


if __name__ == '__main__':
    np.random.seed(42)
    EXPORT = False
    urm, icm, ucm, target_users = build_all_matrices()
    age_ucm = build_age_ucm(urm.shape[0])
    if EXPORT:
        urm_train = urm.tocsr()
        urm_test = None
    else:
        urm_train, urm_test = train_test_split(urm, SplitType.PROBABILISTIC)
    # TOP-POP
    top_pop = TopPopUCM()
    top_pop.fit(urm_train, age_ucm)
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
