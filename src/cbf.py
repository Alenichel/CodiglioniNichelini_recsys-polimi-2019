#!/usr/bin/env python3

import numpy as np
from bayes_opt import BayesianOptimization
from Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from basic_recommenders import TopPopRecommender
from clusterization import get_clusters
from run_utils import set_seed, build_all_matrices, train_test_split, SplitType, export, evaluate, get_cold_users


class ItemCBFKNNRecommender:

    def __init__(self, fallback_recommender=None):
        self.urm = None
        self.icm = None
        self.w_sparse = None
        self.fallback_recommender = fallback_recommender

    def __str__(self):
        return 'Item CBF'

    def fit(self, urm, icm, top_k=50, shrink=100, normalize=True, similarity='tanimoto'):
        self.urm = urm
        self.icm = icm
        similarity_object = Compute_Similarity_Python(self.icm.T, shrink=shrink,
                                                      topK=top_k, normalize=normalize,
                                                      similarity=similarity)
        self.w_sparse = similarity_object.compute_similarity()

    def get_scores(self, user_id, exclude_seen=True):
        user_profile = self.urm[user_id]
        scores = user_profile.dot(self.w_sparse).toarray().ravel()
        if exclude_seen:
            scores = self.filter_seen(user_id, scores)
        return scores

    def recommend(self, user_id, at=None, exclude_seen=True):
        user_profile = self.urm[user_id]
        if user_profile.nnz == 0 and self.fallback_recommender:
            return self.fallback_recommender.recommend(user_id, at=at, exclude_seen=exclude_seen)
        else:
            scores = self.get_scores(user_id, exclude_seen)
            ranking = scores.argsort()[::-1]
            return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.urm.indptr[user_id]
        end_pos = self.urm.indptr[user_id+1]
        user_profile = self.urm.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores


def get_top_user_CBF(urm_train, ucm):
    user_cbf = UserCBFKNNRecommender()
    user_cbf.fit(urm_train, ucm, top_k=492, shrink=211.86, normalize=False, similarity='dice')
    return user_cbf


class UserCBFKNNRecommender:

    def __init__(self):
        self.urm = None
        self.ucm = None
        self.w_sparse = None

    def __str__(self):
        return 'User CBF'

    def fit(self, urm, ucm, top_k=50, shrink=100, normalize=True, similarity='tanimoto'):
        self.urm = urm
        self.ucm = ucm
        similarity_object = Compute_Similarity_Python(self.ucm.T, shrink=shrink,
                                                      topK=top_k, normalize=normalize,
                                                      similarity=similarity)
        self.w_sparse = similarity_object.compute_similarity()

    def get_scores(self, user_id, exclude_seen=True):
        scores = self.w_sparse[user_id, :].dot(self.urm).toarray().ravel()
        if exclude_seen:
            scores = self.filter_seen(user_id, scores)
        return scores

    def recommend(self, user_id, at=None, exclude_seen=True):
        scores = self.get_scores(user_id, exclude_seen)
        ranking = scores.argsort()[::-1]
        return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.urm.indptr[user_id]
        end_pos = self.urm.indptr[user_id+1]
        user_profile = self.urm.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores


class GroupUserCBF:

    def __str__(self):
        return 'Group User CBF'

    def __init__(self, urm_train, ucm, top_k, shrink, normalize, similarity='tanimoto'):
        assert type(top_k) == list
        assert type(shrink) == list
        assert type(normalize) == list
        assert type(similarity) == list
        assert len(top_k) == len(shrink)
        self.default_cbf = get_top_user_CBF(urm_train, ucm)
        self.recommender = list()
        self.clusters, self.user_to_clusters = get_clusters(urm_train, n_cluster=4, max_iter=300, remove_warm=True, return_users_to_cluster=True)
        for n in range(len(self.clusters)):
            user_cbf = UserCBFKNNRecommender()
            user_cbf.fit(urm_train, ucm, top_k=top_k[n], shrink=shrink[n], normalize=normalize[n], similarity=similarity[n])
            self.recommender.append(user_cbf)

    def recommend(self, user_id, at=None, exclude_seen=True):
        try:
            return self.recommender[self.user_to_clusters[user_id]].recommend(user_id, at, exclude_seen)
        except KeyError:
            return self.default_cbf.recommend(user_id, at, exclude_seen)


def tuner():
    urm, icm, ucm, _ = build_all_matrices()
    urm_train, urm_test = train_test_split(urm, SplitType.PROBABILISTIC)

    profile_lengths = np.ediff1d(urm_train.indptr)
    warm_users = np.where(profile_lengths != 0)[0]

    similarities = ['cosine', 'adjusted', 'asymmetric', 'pearson', 'jaccard', 'dice', 'tversky', 'tanimoto']

    pbounds = {
        'top_k': (0, 500),
        'shrink': (0, 500),
        'normalize': (0, 1),
        'similarity': (0, len(similarities))
    }

    def rec_round(top_k, shrink, normalize, similarity):
        top_k = int(top_k)
        normalize = normalize < 0.5
        similarity = similarities[int(similarity)]
        cbf = UserCBFKNNRecommender()
        cbf.fit(urm_train, ucm, top_k=top_k, shrink=shrink, normalize=normalize, similarity=similarity)
        return evaluate(cbf, urm_test, verbose=False, excluded_users=warm_users)['MAP']

    optimizer = BayesianOptimization(f=rec_round, pbounds=pbounds)
    optimizer.maximize(init_points=50, n_iter=250)
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))
    print(optimizer.max)


if __name__ == '__main__':
    set_seed(42)
    #tuner()
    #exit()

    EXPORT = False
    urm, icm, ucm, target_users = build_all_matrices()
    if EXPORT:
        urm_train = urm.tocsr()
        urm_test = None
    else:
        urm_train, urm_test = train_test_split(urm)
    _, warm_users = get_cold_users(urm_train, return_warm=True)
    cbf_rec = UserCBFKNNRecommender()
    cbf_rec.fit(urm_train, ucm, top_k=492, shrink=211.86, normalize=False, similarity='dice')
    clusters = get_clusters(urm_train, n_cluster=4, max_iter=300, remove_warm=True)
    #top_pop = TopPopRecommender()
    #top_pop.fit(urm_train)
    #cbf_rec = top_pop
    to_evaluate = clusters[2]
    to_exclude = list()
    for user_id in range(urm.shape[0]):
        if user_id not in to_evaluate:
            to_exclude.append(user_id)
    '''
    #top_Ks = [224, 93, 481, 101]
    top_Ks = [224, 93, 492, 101]
    #shrinks = [375.68, 370.39, 196.68, 186.31]
    shrinks = [375.68, 370.39, 211.86, 186.31]
    normalizes = [True, True, False, True]
    similarities = ['dice' for i in range(4)]
    cbf_rec = GroupUserCBF(urm_train, ucm, top_Ks, shrinks, normalizes, similarities)'''
    if EXPORT:
        export(target_users, cbf_rec)
    else:
        #evaluate(cbf_rec, urm_test, excluded_users=get_cold_users(urm_train, return_warm=True)[1])
        evaluate(cbf_rec, urm_test, excluded_users=to_exclude)