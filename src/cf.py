#!/usr/bin/env python3

import numpy as np
from Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python


class ItemCFKNNRecommender(object):

    def __init__(self):
        self.urm = None
        self.w_sparse = None
        self.weights = None

    def __create_weights(self):
        weights = []
        num_users = self.urm.shape[0]
        num_items = self.urm.shape[1]
        item_popularity = self.urm.sum(axis=0).squeeze()
        for item_id in range(num_items):
            m_j = item_popularity[0, item_id]       # WARNING: THIS CAN BE 0!!!
            weights.append(np.log(num_users / m_j))
        self.weights = np.array(weights)

    def fit(self, urm, top_k=50, shrink=0, normalize=False, similarity='cosine'):
        self.urm = urm
        similarity_object = Compute_Similarity_Python(self.urm, shrink=shrink,
                                                      topK=top_k, normalize=normalize,
                                                      similarity=similarity)
        self.w_sparse = similarity_object.compute_similarity()
        self.__create_weights()

    def recommend(self, user_id, at=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.urm[user_id]
        scores = user_profile.dot(self.w_sparse).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]


        return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.urm.indptr[user_id]
        end_pos = self.urm.indptr[user_id + 1]

        user_profile = self.urm.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores
