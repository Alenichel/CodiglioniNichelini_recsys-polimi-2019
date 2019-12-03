#!/usr/bin/env python3

from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from run_utils import build_all_matrices, train_test_split, evaluate, export, SplitType


EXPORT = False
urm, icm, target_users = build_all_matrices()
if EXPORT:
    urm_train = urm
    urm_test = None
else:
    urm_train, urm_test = train_test_split(urm, SplitType.LOO)
model = LightFM(loss='warp-kos')
model.fit(urm_train, epochs=1, verbose=True)
print(precision_at_k(model, urm_test, train_interactions=urm_train, k=10).mean())
