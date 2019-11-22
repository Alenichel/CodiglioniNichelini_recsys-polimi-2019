import numpy as np


class TailBoost:

    def __init__(self, urm):
        self.weights = list()
        self.urm = urm
        self.__create_weights()

    def __create_weights(self):
        num_users = self.urm.shape[0]
        num_items = self.urm.shape[1]
        item_popularity = self.urm.sum(axis=0).squeeze()
        for item_id in range(num_items):
            m_j = item_popularity[0, item_id]       # WARNING: THIS CAN BE 0!!!
            if m_j == 0:
                m_j = 1
            self.weights.append(np.log(num_users / m_j))
        self.weights = np.array(self.weights)

    def update_scores(self, scores):
        for i in range(len(scores)):
            scores[i] = scores[i] * self.weights[i]
        return scores

