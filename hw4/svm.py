#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cvxopt import matrix, solvers
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel


class SVM(object):

    def __init__(self, kernel="linear", c=1.0, degree=None, gamma=1.0):
        self.kernel = kernel
        self.c = c

        # This parameter is used for polynomial model.
        self.degree = degree

        # This parameter for Radial kernel. K(x, x') = exp(-gamma * ||x - x'||^2)
        self.gamma = gamma

    def _preprocess_label(self, y):
        # If elements in label aren't either -1 or 1, then change invalid values
        # to meet the SVM.
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
        if self.kernel == "polynomial":
            # K(x, y) = (x^T * y + 1)^n   ('*' is dot product)
            X_ = polynomial_kernel(X, X, degree=self.degree)
            polynomial_matrix = PolynomialFeatures(degree = self.degree)
            X = polynomial_matrix.fit_transform(X)
            # X_ = np.dot(X, X.T)
            P = y.T * X_ * y
        elif self.kernel == "radial":
            X_ = rbf_kernel(X, X, gamma=self.gamma)
            P = y.T * X_ * y
        else:
            X_ = np.dot(X, X.T)
            y_ = np.dot(y, y.T)
            P = X_ * y_
        m, n = X.shape

        minus_i = -1* np.identity(m)
        i = np.identity(m)
        G = np.concatenate((minus_i, i), axis=0).T
        q = -1 * np.ones((1, m))
        c = self.c * (np.ones((m, 1)))
        h = np.concatenate((np.zeros((m, 1)), c), axis=0).T

        b = np.array([0])
        A = y

        self.alpha = np.array(self._qp_solver(P, q, G, h ,A, b)['x'])
        w = np.dot((y * X).T, self.alpha)
        self.w = w

        sp_idx = []
        for idx, val in enumerate(self.alpha):
            if val[0] > 0.001:
                sp_idx.append(idx)
            else:
                val[0] = 0
        X_sp = X[sp_idx]
        y_sp = y[sp_idx]
        w_0 = np.mean(y_sp - np.dot(X_sp, w))
        self.w_0 = w_0
        self.coef_ = (w, w_0, sp_idx)

    def fit(self, X, y):
        y = self._preprocess_label(y)
        self._compute_parameter(X, y)


    def predict(self, X):
        if self.kernel == "polynomial":
            polynomial_matrix = PolynomialFeatures(degree = self.degree)
            X = polynomial_matrix.fit_transform(X)
        else:
            X = X
        f = np.dot(X, self.w) + self.w_0
        for i in f:
            if i[0] >= 0:
                i[0] = 1
            else:
                i[0] = -1
        return f

