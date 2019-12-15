#!/usr/bin/env python3

import numpy as np
from bayes_opt import BayesianOptimization
from Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from run_utils import build_all_matrices, train_test_split, SplitType, export, evaluate


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


def tuner():
    urm, icm, ucm, _ = build_all_matrices()
    urm_train, urm_test = train_test_split(urm, SplitType.PROBABILISTIC)
    pbounds = {'top_k': (0, 500), 'shrink': (0, 500), 'normalize': (0, 1)}

    def rec_round(top_k, shrink, normalize):
        top_k = int(top_k)
        shrink = int(shrink)
        normalize = normalize < 0.5
        cbf = ItemCBFKNNRecommender()
        cbf.fit(urm_train, icm, top_k=top_k, shrink=shrink, normalize=normalize, similarity='tanimoto')
        return evaluate(cbf, urm_test, cython=True, verbose=False)['MAP']

    optimizer = BayesianOptimization(f=rec_round, pbounds=pbounds)
    optimizer.maximize(init_points=10, n_iter=500)
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))
    print(optimizer.max)


if __name__ == '__main__':
    tuner()
    exit()

    EXPORT = False
    urm, icm, ucm, target_users = build_all_matrices()
    if EXPORT:
        urm_train = urm.tocsr()
        urm_test = None
    else:
        urm_train, urm_test = train_test_split(urm, SplitType.LOO_CYTHON)
    cbf_rec = ItemCBFKNNRecommender()
    cbf_rec.fit(urm_train, icm)
    if EXPORT:
        export(target_users, cbf_rec)
    else:
        evaluate(cbf_rec, urm_test)