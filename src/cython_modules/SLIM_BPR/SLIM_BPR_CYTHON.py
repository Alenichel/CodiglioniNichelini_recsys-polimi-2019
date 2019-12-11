#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/09/17

@author: Maurizio Ferrari Dacrema
"""

from Base.Recommender_utils import check_matrix
from Base.BaseSimilarityMatrixRecommender import BaseSimilarityMatrixRecommender
from Base.Recommender_utils import similarityMatrixTopK
from Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from cython_modules.CythonCompiler.run_compile_subprocess import run_compile_subprocess
import os, sys

import numpy as np

def estimate_required_MB(n_items, symmetric):
    requiredMB = 8 * n_items**2 / 1e+06
    if symmetric:
        requiredMB /=2
    return requiredMB


def get_RAM_status():
    try:
        data_list = os.popen('free -t -m').readlines()[1].split()
        tot_m = float(data_list[1])
        used_m = float(data_list[2])
        available_m = float(data_list[6])
    except Exception as exc:
        print("Unable to read memory status: {}".format(str(exc)))
        tot_m, used_m, available_m = None, None, None
    return tot_m, used_m, available_m


class SLIM_BPR(Incremental_Training_Early_Stopping):
    RECOMMENDER_NAME = "SLIM BPR (Cython)"

    def __str__(self):
        return self.RECOMMENDER_NAME

    def __init__(self,
                 verbose = True,
                 free_mem_threshold = 0.5,
                 recompile_cython = False,
                 use_tailboost=False,
                 fallback_recommender=None
                 ):
        self.urm_train = None
        self.n_users = None
        self.n_items = None
        self.W = None
        self.fallback_recommender=fallback_recommender
        self.use_tailboost = use_tailboost

        #super(SLIM_BPR, self).__init__(URM_train)
        #self.free_mem_threshold = free_mem_threshold
        if recompile_cython:
            print("Compiling in Cython")
            self.run_compilation_script()
            print("Compilation Complete")

    def fit(self, urm_train,
            epochs=300,
            positive_threshold_BPR = None,
            train_with_sparse_weights = None,
            symmetric = True,
            random_seed = None,
            batch_size = 1000, lambda_i = 0.0, lambda_j = 0.0, learning_rate = 1e-4, topK = 200,
            sgd_mode='adagrad', gamma=0.995, beta_1=0.9, beta_2=0.999,
            **earlystopping_kwargs):


        # Import compiled module
        from cython_modules.SLIM_BPR.SLIM_BPR_Cython_Epoch import SLIM_BPR_Cython_Epoch
        self.urm_train = urm_train
        self.n_users, self.n_items = self.urm_train.shape
        self.symmetric = symmetric
        self.train_with_sparse_weights = train_with_sparse_weights

        if self.train_with_sparse_weights is None:
            # auto select
            required_m = estimate_required_MB(self.n_items, self.symmetric)
            total_m, _, available_m = get_RAM_status()
            if total_m is not None:
                string = "Automatic selection of fastest train mode. Available RAM is {:.2f} MB ({:.2f}%) of {:.2f} MB, required is {:.2f} MB. ".format(available_m, available_m/total_m*100 , total_m, required_m)
            else:
                string = "Automatic selection of fastest train mode. Unable to get current RAM status, you may be using a non-Linux operating system. "

            if total_m is None or required_m/available_m < self.free_mem_threshold:
                print(string + "Using dense matrix.")
                self.train_with_sparse_weights = False
            else:
                print(string + "Using sparse matrix.")
                self.train_with_sparse_weights = True

        # Select only positive interactions
        URM_train_positive = self.urm_train.copy()

        self.positive_threshold_BPR = positive_threshold_BPR
        self.sgd_mode = sgd_mode
        self.epochs = epochs

        if self.positive_threshold_BPR is not None:
            URM_train_positive.data = URM_train_positive.data >= self.positive_threshold_BPR
            URM_train_positive.eliminate_zeros()

            assert URM_train_positive.nnz > 0, "SLIM_BPR_Cython: URM_train_positive is empty, positive threshold is too high"

        self.cythonEpoch = SLIM_BPR_Cython_Epoch(URM_train_positive,
                                                 train_with_sparse_weights = self.train_with_sparse_weights,
                                                 final_model_sparse_weights = True,
                                                 topK=topK,
                                                 learning_rate=learning_rate,
                                                 li_reg = lambda_i,
                                                 lj_reg = lambda_j,
                                                 batch_size=1,
                                                 symmetric = self.symmetric,
                                                 sgd_mode = sgd_mode,
                                                 verbose = True,
                                                 random_seed = random_seed,
                                                 gamma=gamma,
                                                 beta_1=beta_1,
                                                 beta_2=beta_2)

        if(topK != False and topK<1):
            raise ValueError("TopK not valid. Acceptable values are either False or a positive integer value. Provided value was '{}'".format(topK))
        self.topK = topK
        self.batch_size = batch_size
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j
        self.learning_rate = learning_rate
        self.S_incremental = self.cythonEpoch.get_S()
        self.S_best = self.S_incremental.copy()
        self._train_with_early_stopping(epochs,
                                        algorithm_name = self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)
        self.get_S_incremental_and_set_W()
        self.cythonEpoch._dealloc()
        sys.stdout.flush()




    def _prepare_model_for_validation(self):
        self.get_S_incremental_and_set_W()

    def _update_best_model(self):
        self.S_best = self.S_incremental.copy()

    def _run_epoch(self, num_epoch):
       self.cythonEpoch.epochIteration_Cython()

    def get_S_incremental_and_set_W(self):

        self.S_incremental = self.cythonEpoch.get_S()

        if self.train_with_sparse_weights:
            self.W = self.S_incremental
            self.W = check_matrix(self.W, format='csr')
        else:
            self.W = similarityMatrixTopK(self.S_incremental, k = self.topK)
            self.W = check_matrix(self.W, format='csr')

    def run_compilation_script(self):
        # Run compile script setting the working directory to ensure the compiled file are contained in the
        # appropriate subfolder and not the project root
        file_subfolder = "/SLIM_BPR/Cython"
        file_to_compile_list = ['SLIM_BPR_Cython_Epoch.pyx']
        run_compile_subprocess(file_subfolder, file_to_compile_list)
        print("{}: Compiled module {} in subfolder: {}".format(self.RECOMMENDER_NAME, file_to_compile_list, file_subfolder))
        # Command to run compilation script
        # python compile_script.py SLIM_BPR_Cython_Epoch.pyx build_ext --inplace
        # Command to generate html report
        # cython -a SLIM_BPR_Cython_Epoch.pyx

    def filter_seen(self, user_id, scores):
        start_pos = self.urm_train.indptr[user_id]
        end_pos = self.urm_train.indptr[user_id + 1]
        user_profile = self.urm_train.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores

    def recommend(self, user_id, at=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.urm_train[user_id]
        if self.fallback_recommender and user_profile.nnz == 0:
            return self.fallback_recommender.recommend(user_id, at, exclude_seen)
        scores = user_profile.dot(self.W)
        scores = scores.toarray().ravel()
        if exclude_seen:
            scores = self.filter_seen(user_id, scores)
        if self.use_tailboost:
            scores = self.tb.update_scores(scores)
        ranking = scores.argsort()[::-1]
        return ranking[:at]

