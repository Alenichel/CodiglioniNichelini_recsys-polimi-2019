from run_utils import *
from slim_bpr import *
import pprint as pp
import numpy as np
from evaluation import evaluate_algorithm
from tqdm import tqdm

N_OF_TRIAL = 5

given_i_param = 0.0025
given_j_param = 0.00025
given_learning_rate = 0.05

if __name__ == '__main__':
    print('Using SLIM(Bpr)')
    candidate_i = [float(x) for x in np.arange(0, given_i_param*2, 0.0005)]
    candidate_j = [float(x) for x in np.arange(0, given_j_param*2, 0.00005)]
    candidate_learning_rate = [x for x in np.arange(0, 0.1, 0.01)]

    urm, icm, target_users = build_all_matrices()

    results = list()

    for c_i in candidate_i:
        for c_j in candidate_j:
            median_value = 0
            cumulative_map = 0
            for x in tqdm(range(N_OF_TRIAL)):
                print("Trying with i=%f, j=%f " % (c_i, c_j))
                urm_train, urm_test = train_test_split(urm, SplitType.LOO)
                recommender = SLIM_BPR(c_i, c_j, learning_rate=0.05, fallback_recommender=None,
                                       use_tailboost=False)
                recommender.fit(urm_train, epochs=30)
                round_MAP = float(evaluate(recommender, urm_test)['MAP'])
                cumulative_map += round_MAP
                median_value = cumulative_map / (x+1)

            result = {
                'value': median_value,
                'i': c_i,
                'j': c_j
            }
            pp.pprint(result)
            results.append(result)

    print('MAX map found:\n')
    pp.pprint(results.sort(key=lambda dictionary: dictionary['value'], reverse=True))

