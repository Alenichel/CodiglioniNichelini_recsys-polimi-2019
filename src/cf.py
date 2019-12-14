#!/usr/bin/env python3

import numpy as np
from run_utils import build_all_matrices, train_test_split, SplitType, export, evaluate
from helper import TailBoost
from Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from basic_recommenders import TopPopRecommender
from evaluation import multiple_evaluation
from pprint import pprint as pp
from bayes_opt import BayesianOptimization


class ItemCFKNNRecommender(object):

    def __init__(self, use_tail_boost=False, fallback_recommender=None):
        self.urm = None
        self.w_sparse = None
        self.use_tail_boost = use_tail_boost
        self.tb = None
        self.fallback_recommender = fallback_recommender    # NOTE: This should be already trained

    def __str__(self):
        return 'Item CF'

    def fit(self, urm, top_k=50, shrink=100, normalize=True, similarity='cosine'):
        #print('top_k={0}, shrink={1}, tail_boost={2}, fallback={3}'.format(top_k, shrink, self.use_tail_boost, self.fallback_recommender))
        self.urm = urm.tocsr()
        if self.use_tail_boost:
            self.tb = TailBoost(urm)
        similarity_object = Compute_Similarity_Python(self.urm, shrink=shrink,
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
            return self.fallback_recommender.recommend(user_id, at, exclude_seen)
        else:
            scores = self.get_scores(user_id, exclude_seen)
            if self.use_tail_boost:
                scores = self.tb.update_scores(scores)
            ranking = scores.argsort()[::-1]
            return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.urm.indptr[user_id]
        end_pos = self.urm.indptr[user_id + 1]
        user_profile = self.urm.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores


class UserCFKNNRecommender(object):

    def __init__(self, use_tail_boost=False, fallback_recommender=None):
        self.urm = None
        self.w_sparse = None
        self.use_tail_boost = use_tail_boost
        self.tb = None
        self.fallback_recommender = fallback_recommender    # NOTE: This should be already trained

    def __str__(self):
        return 'User CF'

    def fit(self, urm, top_k=50, shrink=100, normalize=True, similarity='cosine'):
        #print('top_k={0}, shrink={1}, tail_boost={2}, fallback={3}'.format(top_k, shrink, self.use_tail_boost, self.fallback_recommender))
        self.urm = urm.tocsr()
        if self.use_tail_boost:
            self.tb = TailBoost(urm)
        similarity_object = Compute_Similarity_Python(self.urm.T, shrink=shrink,
                                                      topK=top_k, normalize=normalize,
                                                      similarity=similarity)
        self.w_sparse = similarity_object.compute_similarity()

    def get_scores(self, user_id, exclude_seen=True):
        scores = self.w_sparse[user_id, :].dot(self.urm).toarray().ravel()
        if exclude_seen:
            scores = self.filter_seen(user_id, scores)
        return scores

    def recommend(self, user_id, at=None, exclude_seen=True):
        user_profile = self.urm[user_id]
        if user_profile.nnz == 0 and self.fallback_recommender:
            return self.fallback_recommender.recommend(user_id, at, exclude_seen)
        else:
            scores = self.get_scores(user_id, exclude_seen)
            if self.use_tail_boost:
                scores = self.tb.update_scores(scores)
            ranking = scores.argsort()[::-1]
            return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.urm.indptr[user_id]
        end_pos = self.urm.indptr[user_id + 1]
        user_profile = self.urm.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores


def tuner():
    urm, icm, ucm, target_users = build_all_matrices()
    urm_train, urm_test = train_test_split(urm, SplitType.LOO_CYTHON)
    top_pop = TopPopRecommender()
    top_pop.fit(urm_train)
    pbounds = {'top_k': (0, 500), 'shrink': (0, 500), 'normalize': (0, 1)}

    def rec_round(top_k, shrink, normalize):
        top_k = int(top_k)
        shrink = int(shrink)
        normalize = normalize < 0.5
        cf = ItemCFKNNRecommender(fallback_recommender=top_pop)
        cf.fit(urm_train, top_k=top_k, shrink=shrink, normalize=normalize, similarity='tanimoto')
        return evaluate(cf, urm_test, cython=True, verbose=False)['MAP']

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
    top_pop = TopPopRecommender()
    top_pop.fit(urm_train)
    cf = ItemCFKNNRecommender(fallback_recommender=top_pop)
    cf.fit(urm_train, top_k=5, shrink=20, similarity='tanimoto')
    if EXPORT:
        export(target_users, cf)
    else:
        evaluate(cf, urm_test, cython=True)

"""
if __name__ == '__main__':
    results = []
    for n in range(25):
        top_k = np.random.randint(0,1000)
        shrink = np.random.randint(0,1000)
        similarity = 'tanimoto'
        results.append(multiple_evaluation(
            ItemCFKNNRecommender(fallback_recommender=TopPopRecommender()), [top_k, shrink, True, similarity],
            round=5, use_group_evaluation=True))
        results.sort(key=lambda dictionary: dictionary['median_value'], reverse=True)
    pp(results)
"""