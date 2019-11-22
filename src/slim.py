#!/usr/bin/env python3

import numpy as np
import scipy.sparse as sps
import time
from Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python


class SLIM_BPR_Recommender:
    """ SLIM_BPR recommender with cosine similarity and no shrinkage"""

    def sampleTriplet(self):

        # By randomly selecting a user in this way we could end up
        # with a user with no interactions
        # user_id = np.random.randint(0, n_users)

        user_id = np.random.choice(self.eligibleUsers)

        # Get user seen items and choose one
        userSeenItems = self.urm[user_id, :].indices
        pos_item_id = np.random.choice(userSeenItems)

        negItemSelected = False

        # It's faster to just try again then to build a mapping of the non-seen items
        neg_item_id = None
        while (not negItemSelected):
            neg_item_id = np.random.randint(0, self.n_items)

            if (neg_item_id not in userSeenItems):
                negItemSelected = True

        return user_id, pos_item_id, neg_item_id

    def epochIteration(self):

        # Get number of available interactions
        numPositiveIteractions = int(self.urm.nnz * 0.01)

        start_time_epoch = time.time()
        start_time_batch = time.time()

        # Uniform user sampling without replacement
        for num_sample in range(numPositiveIteractions):

            # Sample
            user_id, positive_item_id, negative_item_id = self.sampleTriplet()

            userSeenItems = self.urm[user_id, :].indices

            # Prediction
            x_i = self.similarity_matrix[positive_item_id, userSeenItems].sum()
            x_j = self.similarity_matrix[negative_item_id, userSeenItems].sum()

            # Gradient
            x_ij = x_i - x_j

            gradient = 1 / (1 + np.exp(x_ij))

            # Update
            self.similarity_matrix[positive_item_id, userSeenItems] += self.learning_rate * gradient
            self.similarity_matrix[positive_item_id, positive_item_id] = 0

            self.similarity_matrix[negative_item_id, userSeenItems] -= self.learning_rate * gradient
            self.similarity_matrix[negative_item_id, negative_item_id] = 0

            if (time.time() - start_time_batch >= 30 or num_sample == numPositiveIteractions - 1):
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Sample per second: {:.0f}".format(
                    num_sample,
                    100.0 * float(num_sample) / numPositiveIteractions,
                    time.time() - start_time_batch,
                    float(num_sample) / (time.time() - start_time_epoch)))

                start_time_batch = time.time()

    def __create_weights(self):
        weights = []
        num_users = self.urm.shape[0]
        num_items = self.urm.shape[1]
        item_popularity = self.urm.sum(axis=0).squeeze()
        for item_id in range(num_items):
            m_j = item_popularity[0, item_id]       # WARNING: THIS CAN BE 0!!!
            if m_j == 0:
                m_j = 1
            weights.append(np.log(num_users / m_j))
        self.weights = np.array(weights)

    def fit(self, urm, learning_rate=0.01, epochs=10):

        self.urm = urm.tocsr()
        self.n_users = self.urm.shape[0]
        self.n_items = self.urm.shape[1]
        self.similarity_matrix = np.zeros((self.n_items, self.n_items))
        self.__create_weights()

        # Extract users having at least one interaction to choose from
        self.eligibleUsers = []

        for user_id in range(self.n_users):

            start_pos = self.urm.indptr[user_id]
            end_pos = self.urm.indptr[user_id + 1]

            if len(self.urm.indices[start_pos:end_pos]) > 0:
                self.eligibleUsers.append(user_id)

        self.learning_rate = learning_rate
        self.epochs = epochs

        for numEpoch in range(self.epochs):
            print('Epoch {0}'.format(numEpoch))
            self.epochIteration()

        self.similarity_matrix = self.similarity_matrix.T

        #self.similarity_matrix = similarityMatrixTopK(self.similarity_matrix, k=100)
        similarity_object = Compute_Similarity_Python(self.urm, topK=100, shrink=100, normalize=True)
        self.similarity_matrix = similarity_object.compute_similarity()

    def recommend(self, user_id, at=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.urm[user_id]
        scores = user_profile.dot(self.similarity_matrix).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        for i in range(len(scores)):
            scores[i] = scores[i] * self.weights[i]
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):

        start_pos = self.urm.indptr[user_id]
        end_pos = self.urm.indptr[user_id + 1]

        user_profile = self.urm.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores


#def similarityMatrixTopK(item_weights, forceSparseOutput = True, k=100, verbose = False, inplace=True):




