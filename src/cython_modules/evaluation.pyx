import numpy as np
from tqdm import trange
cimport cython
cimport numpy as np


cdef precision(np.ndarray is_relevant, np.ndarray relevant_items):
    cdef double precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)
    return precision_score


cdef recall(np.ndarray is_relevant, np.ndarray relevant_items):
    cdef double recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]
    return recall_score


cdef MAP(np.ndarray is_relevant, np.ndarray relevant_items):
    # Cumulative sum: precision at 1, at 2, at 3 ...
    cdef np.ndarray p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
    cdef double map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])
    return map_score


@cython.boundscheck(False)
@cython.wraparound(False)
def evaluate_cython(recommender_object, urm_test, int at=10):
    cdef double cumulative_precision = 0.0
    cdef double cumulative_recall = 0.0
    cdef double cumulative_MAP = 0.0
    cdef int num_eval = 0
    urm_test = urm_test.tocsr()
    cdef int n_users = urm_test.shape[0]
    cdef int start_pos, end_pos
    cdef np.ndarray relevant_items
    cdef np.ndarray recommended_items
    cdef np.ndarray is_relevant
    for user_id in trange(n_users, desc='Evaluation'):
        start_pos = urm_test.indptr[user_id]
        end_pos = urm_test.indptr[user_id+1]
        if end_pos - start_pos > 0:
            relevant_items = urm_test.indices[start_pos:end_pos]
            recommended_items = recommender_object.recommend(user_id, at=at)
            num_eval += 1
            is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
            cumulative_precision += precision(is_relevant, relevant_items)
            cumulative_recall += recall(is_relevant, relevant_items)
            cumulative_MAP += MAP(is_relevant, relevant_items)
    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_MAP /= num_eval
    print('Recommender performance is:')
    print('    Precision = {:.5f}'.format(cumulative_precision))
    print('    Recall    = {:.5f}'.format(cumulative_recall))
    print('    MAP       = {:.5f}'.format(cumulative_MAP))
    result_dict = {
        "precision": cumulative_precision,
        "recall": cumulative_recall,
        "MAP": cumulative_MAP,
    }
    return result_dict