#!/usr/bin/env python3

import numpy as np
import scipy.sparse as sps
from sklearn.linear_model import ElasticNet
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from tqdm import trange
from run_utils import build_all_matrices, train_test_split, SplitType, export, evaluate


class SLIMElasticNetRecommender:

    def __init__(self):
        self.urm_train = None
        self.model = None
        self.W_sparse = None

    def __str__(self):
        return 'SLIM ElasticNet'

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, urm_train, l1_penalty=0.01, l2_penalty=0.01, positive_only=True, topK=100):
        self.urm_train = urm_train
        if l1_penalty + l2_penalty != 0:
            l1_ratio = l1_penalty / (l1_penalty + l2_penalty)
        else:
            print("SLIM_ElasticNet: l1_penalty+l2_penalty cannot be equal to zero, setting the ratio l1/(l1+l2) to 1.0")
            l1_ratio = 1.0
        self.model = ElasticNet(alpha=1e-4,
                                l1_ratio=l1_ratio,
                                positive=positive_only,
                                fit_intercept=False,
                                copy_X=False,
                                precompute=True,
                                selection='random',
                                max_iter=100,
                                tol=1e-4)
        urm_train = sps.csc_matrix(self.urm_train)
        n_items = urm_train.shape[1]
        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000
        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)
        numCells = 0
        # fit each item's factors sequentially (not in parallel)
        for currentItem in trange(n_items, desc='Fitting'):
            # get the target column
            y = urm_train[:, currentItem].toarray()
            if y.sum() == 0:
                continue

            # set the j-th column of X to zero
            start_pos = urm_train.indptr[currentItem]
            end_pos = urm_train.indptr[currentItem + 1]

            current_item_data_backup = urm_train.data[start_pos: end_pos].copy()
            urm_train.data[start_pos: end_pos] = 0.0
            # fit one ElasticNet model per column
            self.model.fit(urm_train, y)

            # self.model.coef_ contains the coefficient of the ElasticNet model
            # let's keep only the non-zero values

            # Select topK values
            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index

            nonzero_model_coef_index = self.model.sparse_coef_.indices
            nonzero_model_coef_value = self.model.sparse_coef_.data

            local_topK = min(len(nonzero_model_coef_value) - 1, topK)

            relevant_items_partition = (-nonzero_model_coef_value).argpartition(local_topK)[0:local_topK]
            relevant_items_partition_sorting = np.argsort(-nonzero_model_coef_value[relevant_items_partition])
            ranking = relevant_items_partition[relevant_items_partition_sorting]

            for index in range(len(ranking)):

                if numCells == len(rows):
                    rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                    cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                    values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))

                rows[numCells] = nonzero_model_coef_index[ranking[index]]
                cols[numCells] = currentItem
                values[numCells] = nonzero_model_coef_value[ranking[index]]

                numCells += 1

            # finally, replace the original values of the j-th column
            urm_train.data[start_pos:end_pos] = current_item_data_backup
        # generate the sparse weight matrix
        self.W_sparse = sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])),
                                       shape=(n_items, n_items), dtype=np.float32)

    def get_scores(self, user_id, exclude_seen=True):
        user_profile = self.urm_train[user_id]
        scores = user_profile.dot(self.W_sparse).toarray().ravel()
        if exclude_seen:
            scores = self.filter_seen(user_id, scores)
        return scores

    def recommend(self, user_id, at=None, exclude_seen=True):
        #user_profile = self.urm_train[user_id]
        scores = self.get_scores(user_id, exclude_seen)
        ranking = scores.argsort()[::-1]
        return ranking[:at]

    def filter_seen(self, user_id, scores):

        start_pos = self.urm_train.indptr[user_id]
        end_pos = self.urm_train.indptr[user_id + 1]

        user_profile = self.urm_train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores


if __name__ == '__main__':
    EXPORT = False
    urm, icm, ucm, target_users = build_all_matrices()
    if EXPORT:
        urm_train = urm.tocsr()
        urm_test = None
    else:
        urm_train, urm_test = train_test_split(urm, SplitType.PROBABILISTIC)
    rec = SLIMElasticNetRecommender()
    rec.fit(urm_train)
    if EXPORT:
        export(target_users, rec)
    else:
        evaluate(rec, urm_test)