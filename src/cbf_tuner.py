from run_utils import build_urm, train_test_split
from cf import ItemCFKNNRecommender
from enum import Enum
from csv_utils import load_csv, export_csv
from evaluation import evaluate_algorithm
import numpy as np
import scipy.sparse as sps
import pprint as pp

class SimilarityFunction(Enum):
    COSINE = "cosine"
    JACCARD = "jaccard"
    TANIMOTO = "tanimoto"
    ADJUSTED_COSINE = "adjusted"
    EUCLIDEAN = "euclidean"


if __name__ == '__main__':
    print('Using Collaborative Filtering (content-based)')
    recommender = ItemCBFKNNRecommender()
    results = []
    for parameter in results['list']:
        parameter['MAP'] = t.evaluate(parameter['top_k'], parameter['shrink'], parameter['similarity_function'])
    results.sort(key=lambda dictionary: dictionary['MAP'], reverse=True)
    pp.pprint(results)