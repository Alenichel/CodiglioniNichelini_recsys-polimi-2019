#!/usr/bin/env python3

import numpy as np
from helper import TailBoost
from Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python


class ItemCFKNNRecommender(object):

    def __init__(self):
        self.urm = None
        self.w_sparse = None
        self.tb = None

    def fit(self, urm, top_k=50, shrink=100, normalize=True, similarity='cosine'):
        self.urm = urm.tocsr()
        self.tb = TailBoost(urm)
        similarity_object = Compute_Similarity_Python(self.urm, shrink=shrink,
                                                      topK=top_k, normalize=normalize,
                                                      similarity=similarity)
        self.w_sparse = similarity_object.compute_similarity()

    def recommend(self, user_id, at=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.urm[user_id]
        scores = user_profile.dot(self.w_sparse).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        scores = self.tb.update_scores(scores)

        ranking = scores.argsort()[::-1]                # PURE MAGIC FUNCTION

        return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.urm.indptr[user_id]
        end_pos = self.urm.indptr[user_id + 1]

        user_profile = self.urm.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores
