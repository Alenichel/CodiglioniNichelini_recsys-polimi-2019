#!/usr/bin/env python3


import numpy as np
import lightgbm as lgb
from run_utils import build_all_matrices, train_test_split, SplitType, evaluate, export


if __name__ == '__main__':
    EXPORT = False
    np.random.seed(42)
    urm, icm, ucm, target_users = build_all_matrices()
    if EXPORT:
        urm_train = urm.tocsr()
        urm_test = None
    else:
        urm_train, urm_test = train_test_split(urm, SplitType.PROBABILISTIC)

    urm_train = urm_train.astype(float)
    urm_test = urm_test.astype(float)

    train_data = lgb.Dataset(urm_train, params={'group_column': 'query=0'})
    test_data = train_data.create_valid(urm_test)
    query = [1 for _ in range(urm.shape[0])]

    gbm = lgb.LGBMRanker()
    gbm.fit(train_data, np.ones(shape=urm.shape), group=urm_train)

    #params = {
    #    'objective': 'lambdarank',
    #}

    #bst = lgb.train(params, train_data, valid_sets=[test_data], early_stopping_rounds=5)
    #print(bst.predict(target_users))

    #if EXPORT:
    #    export(target_users, cf)
    #else:
    #    evaluate(cf, urm_test, cython=True)