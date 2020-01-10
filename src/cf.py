#!/usr/bin/env python3

import numpy as np
from tqdm import trange

from run_utils import set_seed, build_all_matrices, train_test_split, SplitType, export, evaluate, multiple_splitting
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

    def fit(self, urm, top_k=50, shrink=100, normalize=True, similarity='tanimoto'):
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

    def fit(self, urm, top_k=50, shrink=100, normalize=True, similarity='tanimoto'):
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


def check_best(bests):
    assert type(bests) == list
    trains, tests, _ = multiple_splitting()

    tops = list()
    cfs = list()

    for best in bests:
        top_k = int(best['params']['top_k'])
        shrink = best['params']['shrink']
        similarity = 'jaccard' if best['params']['similarity'] < 0.5 else 'tanimoto'
        normalize = best['params']['normalize'] < 0.5
        cumulative_MAP = 0
        for n in trange(len(trains)):
            cf = ItemCFKNNRecommender()
            cf.fit(trains[n], top_k=top_k, shrink=shrink, normalize=normalize, similarity=similarity)
            cumulative_MAP += evaluate(cf, tests[n], cython=True, verbose=False)['MAP']
        averageMAP = cumulative_MAP / len(trains)
        best['AVG_MAP'] = averageMAP

    for best in bests:
        print(best)

    return bests


def tuner():
    set_seed(42)
    urm, icm, ucm, target_users = build_all_matrices()
    urm_train, urm_test = train_test_split(urm, SplitType.PROBABILISTIC)
    #top_pop = TopPopRecommender()
    #top_pop.fit(urm_train)
    similarities = ['jaccard', 'tanimoto']
    pbounds = {'top_k': (0, 1000), 'shrink': (0, 1000), 'normalize': (0, 1), 'similarity': (0, 1)}

    def rec_round(top_k, shrink, normalize, similarity):
        top_k = int(top_k)
        shrink = int(shrink)
        normalize = normalize < 0.5
        similarity = similarities[0] if similarity < 0.5 else similarities[1]
        cf = ItemCFKNNRecommender()
        cf.fit(urm_train, top_k=top_k, shrink=shrink, normalize=normalize, similarity=similarity)
        return evaluate(cf, urm_test, cython=True, verbose=False)['MAP']

    optimizer = BayesianOptimization(f=rec_round, pbounds=pbounds)
    optimizer.maximize(init_points=2, n_iter=0)
    #for i, res in enumerate(optimizer.res):
    #    print("Iteration {}: \n\t{}".format(i, res))
    #print(optimizer.max)

    opt_results = optimizer.res
    opt_results.sort(key= lambda dic: dic['target'], reverse=True)
    check_best(opt_results[:5])


if __name__ == '__main__':
    tuner()
    exit()

    EXPORT = False
    set_seed(42)
    urm, icm, ucm, target_users = build_all_matrices()
    if EXPORT:
        urm_train = urm.tocsr()
        urm_test = None
    else:
        urm_train, urm_test = train_test_split(urm, SplitType.PROBABILISTIC)
    top_pop = TopPopRecommender()
    top_pop.fit(urm_train)
    cf = ItemCFKNNRecommender(fallback_recommender=top_pop)
    cf.fit(urm_train, top_k=690, shrink=66, normalize=False, similarity='tanimoto')
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