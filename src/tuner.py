
from run import DataFiles, Runner
from cf import ItemCFKNNRecommender
from enum import Enum
from load_export_csv import load_csv, export_csv
from evaluation import evaluate_algorithm
import numpy as np
import scipy.sparse as sps
import pprint as pp

class SimilarityFunction(Enum):
    COSINE = "cosine"
    """PEARSON = "pearson"
    JACCARD = "jaccard"
    TANIMOTO = "tanimoto"
    ADJUSTED_COSINE = "adjusted"
    EUCLIDEAN = "euclidean"""""


DEFAULT_TOP_K = 50
DEFAULT_SHRINK = 100

top_k_parameter_pool = []
shrink_parameter_pool = []


class Tuner:

    def __init__(self, step=5):
        self.urm = None
        self.icm = None
        self.target_users = None

        for n in range(5, DEFAULT_TOP_K * 2 + 1, step):
            top_k_parameter_pool.append(n)

        for n in range(5, DEFAULT_SHRINK * 2 + 1, step):
            shrink_parameter_pool.append(n)

    def prepare_data(self):
        urm_data = load_csv(DataFiles.TRAIN)
        urm_data = [[int(row[i]) if i <= 1 else int(float(row[i])) for i in range(len(row))] for row in urm_data]
        users, items, ratings = map(np.array, zip(*urm_data))
        self.urm = sps.coo_matrix((ratings, (users, items)))
        price_icm_items, price_icm_features, price_icm_values = Runner.load_icm_csv(DataFiles.ICM_PRICE)
        asset_icm_items, asset_icm_features, asset_icm_values = Runner.load_icm_csv(DataFiles.ICM_ASSET)
        asset_icm_features += 1     # asset had feature number 0, now it's 1
        subclass_icm_items, subclass_icm_features, subclass_icm_values = Runner.load_icm_csv(DataFiles.ICM_SUBCLASS)
        subclass_icm_features += 1  # subclass numbers starts from 1, so we add 1 to make room for price and asset
        icm_items = np.concatenate((price_icm_items, asset_icm_items, subclass_icm_items))
        icm_features = np.concatenate((price_icm_features, asset_icm_features, subclass_icm_features))
        icm_values = np.concatenate((price_icm_values, asset_icm_values, subclass_icm_values))
        self.icm = sps.csr_matrix((icm_values, (icm_items, icm_features)))
        self.target_users = load_csv(DataFiles.TARGET_USERS_TEST)
        self.target_users = [int(x[0]) for x in self.target_users]
        return Runner.train_test_loo_split(self.urm)

    def run_tuner(self):
        list_of_parameters = list()

        parameter = {
            'MAP': 0,
            'similarity_function': '',
            'top_k': 0,
            'shrink': 0
        }

        tuning_results = {
            'top_MAP': 0,
            'bottom_MAP': 0,
            'list': list_of_parameters
        }

        top_param = dict()
        for sim_func in SimilarityFunction:
            for top_k_param in top_k_parameter_pool:
                for shrink_param in shrink_parameter_pool:
                    print('\n\n\nEvaluating CF with %s similarity, %d top_k, %d shrink' % (sim_func.value, top_k_param, shrink_param))
                    print('Preparing data...')
                    urm_train, urm_test = self.prepare_data()
                    print('OK\nFitting...')
                    recommender.fit(urm_train)
                    print('Evaluating...')
                    round_MAP = evaluate_algorithm(urm_test, recommender)['MAP']
                    round_param = dict()
                    round_param['MAP'] = round_MAP
                    round_param['similarity_function'] = sim_func.value
                    round_param['top_k'] = top_k_param
                    round_param['shrink'] = shrink_param

                    if round_MAP > tuning_results['top_MAP']:
                        tuning_results['top_MAP'] = round_MAP
                        top_param = round_param
                    list_of_parameters.append(round_param)

        print("Obtained %d with the following parameters %s, %d, %f"
              % (top_param['MAP'], top_param['similarity_function'], top_param['top_k'], top_param['shrink']))

        list_of_parameters.sort(key=lambda dictionary: dictionary['MAP'], reverse=True)
        top_ten = list_of_parameters[:10]
        tuning_results['bottom_MAP'] = top_ten[-1]['MAP']

        print('\nGot also: \n')
        pp.pprint(tuning_results)
        return tuning_results

    def evaluate(self, top_k, shrink, similarity='cosine', round=10):
        median_value = int()
        cumulative_map = 0
        for n in range(round):
            print('\n\n\nEvaluating CF with %s similarity, %d top_k, %d shrink (ROUND %d)'
                  % (similarity, top_k, shrink, n))
            print('Preparing data...')
            urm_train, urm_test = self.prepare_data()
            print('OK\nFitting...')
            recommender.fit(urm_train)
            print('Evaluating...')
            cumulative_map += evaluate_algorithm(urm_test, recommender)['MAP']
        median_value = cumulative_map / round
        print('The median value of MAP after %d round is: %f' % (round, median_value))
        return median_value


if __name__ == '__main__':
    print('Using Collaborative Filtering (item-based)')
    recommender = ItemCFKNNRecommender()
    t = Tuner()
    results = t.run_tuner()
    for parameter in results['list']:
        parameter['MAP'] = t.evaluate(parameter['top_k'], parameter['shrink'], parameter['similarity_function'])
    results.sort(key=lambda dictionary: dictionary['MAP'], reverse=True)
    pp.pprint(results)