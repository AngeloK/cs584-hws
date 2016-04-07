#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cvxopt import matrix, solvers
import numpy as np

class SVM(object):

    def __init__(self):
        pass

    def _preprocess_label(self, y):
        from collections import Counter
        # Using build-in module to compute the count of each class.
        class_count = Counter(y)
        self._class_list = [k for k in class_count.keys()]
        if len(self._class_list) > 2:
            raise ValueError("Only two-label dataset can be used.")
        y = y.reshape((y.shape[0], 1))
        if sum(self._class_list) != 0:
            for i in y:
                if i[0] != 1:
                    i[0] = -1
        self._class_list = [-1, 1]
        return y

    def _qp_solver(self, *parameters):
        new_para = []
        for p in parameters:
            p = matrix(p.tolist(), tc="d")
            new_para.append(p)

        for p in new_para:
            print p.size


        s = solvers.qp(*new_para)
        return s

    def _compute_parameter(self, X, y):
        m, n = X.shape
        X_ = np.dot(X, X.T)
        y_ = np.dot(y, y.T)

        P = X_ * y_
        minus_i = -1* np.identity(m)
        i = np.identity(m)
        G = np.concatenate((minus_i, i), axis=0).T
        q = -1 * np.ones((1, m))
        c = (np.ones((m, 1)))
        h = np.concatenate((np.zeros((m, 1)), c), axis=0).T

        b = np.array([0])
        A = y

        self.alpha = np.array(self._qp_solver(P, q, G, h ,A, b)['x'])
        # print np.dot(self.alpha.T, y)

        w = np.dot((y * X).T, self.alpha)

        sp_idx = []
        for idx, val in enumerate(self.alpha):
            if val[0] > 0.001:
                sp_idx.append(idx)

        X_sp = X[sp_idx]
        y_sp = y[sp_idx]
        w_0 = np.mean(y_sp - np.dot(X_sp, w))
        self.coef_ = (w, w_0, sp_idx)

    def fit(self, X, y):
        y = self._preprocess_label(y)
        self._compute_parameter(X, y)
