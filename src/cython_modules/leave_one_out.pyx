import numpy as np
import scipy.sparse as sps
from tqdm import trange
cimport cython
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def train_test_loo_split(urm):
    print('Using LeaveOneOut (Cython)')
    urm_train = urm.tocsr().copy()
    cdef int num_users, num_items
    num_users, num_items = urm_train.shape
    urm_test = sps.lil_matrix((num_users, num_items), dtype=int)
    cdef int start_pos, end_pos
    cdef np.ndarray user_profile
    cdef int item_id
    for user_id in trange(num_users, desc='LeaveOneOut'):
        start_pos = urm_train.indptr[user_id]
        end_pos = urm_train.indptr[user_id + 1]
        user_profile = urm_train.indices[start_pos:end_pos]
        if user_profile.size > 0:
            item_id = np.random.choice(user_profile, 1)
            urm_train[user_id, item_id] = 0
            urm_test[user_id, item_id] = 1
    urm_test = sps.csr_matrix(urm_test, dtype=int, shape=urm.shape)
    urm_train.eliminate_zeros()
    urm_test.eliminate_zeros()
    return urm_train, urm_test