#!/usr/bin/env python3

import numpy as np
from run_utils import build_all_matrices, train_test_split, evaluate, export, SplitType
from helper import TailBoost
from Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from cbf import ItemCBFKNNRecommender


class ItemCFKNNRecommender(object):

    def __init__(self, use_tail_boost=False, fallback_recommender=None):
        self.urm = None
        self.w_sparse = None
        self.use_tail_boost = use_tail_boost
        self.tb = None
        self.fallback_recommender = fallback_recommender    # NOTE: This should be already trained

    def fit(self, urm, top_k, shrink, similarity, normalize=True):
        print('top_k={0}, shrink={1}, tail_boost={2}'.format(top_k, shrink, self.use_tail_boost))
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


if __name__ == '__main__':
    EXPORT = True
    urm, icm, target_users = build_all_matrices()
    if EXPORT:
        urm_train = urm.tocsr()
        urm_test = None
    else:
        urm_train, urm_test = train_test_split(urm, SplitType.LOO)
    cbf_rec = ItemCBFKNNRecommender()
    cbf_rec.fit(urm_train, icm, top_k=5, shrink=20, similarity='tanimoto')
    cf_rec = ItemCFKNNRecommender(fallback_recommender=cbf_rec)
    cf_rec.fit(urm_train, top_k=5, shrink=20, similarity='tanimoto')
    if EXPORT:
        export(target_users, cf_rec)
    else:
        evaluate(cf_rec, urm_test)
