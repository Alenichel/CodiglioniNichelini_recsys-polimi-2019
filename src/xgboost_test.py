#!/usr/bin/env python3

import numpy as np
import xgboost as xgb
import pandas as pd
from tqdm import trange
from cf import ItemCFKNNRecommender
from basic_recommenders import TopPopRecommender
from run_utils import build_all_matrices, train_test_split, evaluate, export, SplitType


class XGBoostedRecommender:

    def __init__(self):
        self.train_df = None
        self.model = None
        self.preds = None

    def fit(self, train_df, train_labels):
        self.train_df = train_df
        num_round = 20  # the number of training iterations (number of trees)
        self.model = xgb.XGBRanker()
        self.model.fit(train_df, train_labels, group=[len(train_df[train_df['user_id'] == user_id]) for user_id in range(urm.shape[0])], verbose=True)
        self.preds = self.model.predict(train_df)

    def recommend(self, user_id, at=None, exclude_seen=True):
        user_interactions = self.train_df[self.train_df['user_id'] == user_id]
        user_predictions = self.preds[user_interactions.index]
        results = dict()
        for i, idx in enumerate(user_interactions.index):
            results[idx] = user_predictions[i]
        results = sorted(results.keys(), key=lambda k: results[k])
        return list(map(lambda x: user_interactions['item_id'][x], results))[:at]


if __name__ == '__main__':
    np.random.seed(42)
    EXPORT = False
    urm, icm, ucm, target_users = build_all_matrices()
    n_users, n_items = urm.shape
    if EXPORT:
        urm_train = urm.tocsr()
        urm_test = None
    else:
        urm_train, urm_test = train_test_split(urm, SplitType.PROBABILISTIC)

    top_pop = TopPopRecommender()
    top_pop.fit(urm_train)
    cf = ItemCFKNNRecommender(fallback_recommender=top_pop)
    cf.fit(urm_train, top_k=4, shrink=34, normalize=False, similarity='jaccard')

    print('Building train DataFrame...')

    item_popularities = np.array(urm.sum(axis=0)).squeeze()
    profile_lengths_per_user = np.ediff1d(urm.indptr)

    cutoff = 20
    items = []
    users = []

    for user_id in trange(n_users, desc='Train set'):
        recommendations = cf.recommend(user_id, at=cutoff)
        items.extend(recommendations)
        users.extend([user_id for _ in range(len(recommendations))])
    pop_scores = [item_popularities[item_id] for item_id in items]
    profile_lengths = [profile_lengths_per_user[user_id] for user_id in users]
    train_labels = list(np.ones(len(users), dtype=int))

    ones_len = len(train_labels)
    for _ in trange(ones_len, desc='Balancing train set'):
        while True:
            user_id = np.random.randint(n_users)
            item_id = np.random.randint(n_items)
            if urm_train[user_id, item_id] == 0:
                break
        users.append(user_id)
        items.append(item_id)
        pop_scores.append(item_popularities[item_id])
        profile_lengths.append(profile_lengths_per_user[user_id])
        train_labels.append(0)

    train_df = pd.DataFrame({
        'user_id': users,
        'item_id': items,
        'item_popularity': pop_scores,
        'profile_length': profile_lengths
    })

    xgb_rec = XGBoostedRecommender()
    xgb_rec.fit(train_df, train_labels)

    if EXPORT:
        export(target_users, xgb_rec)
    else:
        evaluate(xgb_rec, urm_test)

