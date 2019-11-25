#!/usr/bin/env python3

import numpy as np
from helper import TailBoost
from Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from basic_recommenders import TopPopRecommender


class ItemCFKNNRecommender(object):

    def __init__(self, use_tail_boost=False):
        self.urm = None
        self.w_sparse = None
        self.use_tail_boost = use_tail_boost
        self.tb = None
        self.fallback_recommender = TopPopRecommender()

    def fit(self, urm, top_k=45, shrink=105, normalize=True, similarity='cosine'):
        self.urm = urm.tocsr()
        if self.use_tail_boost:
            print('Tail Boost: ON')
            self.tb = TailBoost(urm)
        else:
            print('Tail Boost: OFF')
        similarity_object = Compute_Similarity_Python(self.urm, shrink=shrink,
                                                      topK=top_k, normalize=normalize,
                                                      similarity=similarity)
        self.w_sparse = similarity_object.compute_similarity()
        self.fallback_recommender.fit(urm)

    def recommend(self, user_id, at=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.urm[user_id]
        """if user_profile.nnz == 0:
            return self.fallback_recommender.recommend(user_id, at, exclude_seen)
        else:"""
        scores = user_profile.dot(self.w_sparse).toarray().ravel()
        if exclude_seen:
            scores = self.filter_seen(user_id, scores)
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
