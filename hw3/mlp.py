#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from lgregression import sigmoid, LogisticRegression


class MLP(object):

    def __init__(self, output=1):
        self.output = output
        self.attr_count = 0
        self.leaning_rate = 0

    def _initialize_parameter(self, X):
        self.attr_count = X.shape[1]
        self.parameter_w = np.ones((self.attr_count, self.attr_count-1))
        self.parameter_v = np.ones((self.attr_count, 1))

    def hidden_layer(self, X):
        Z = sigmoid(np.dot(X, self.parameter_w))
        return Z

    def fit(self, X, y, learning_rate, iteration=100):
        self._initialize_parameter(X)
        self.learning_rate = learning_rate

        for i in range(iteration):
            Z = hidden_layer(X)
            y_hat = sigmoid(self.parameter_v.T, Z)
            print y_hat.T.shape
            print y.shape
            self.parameter_v -= self.learning_rate * (
                np.sum((y_hat.T - y.T) * Z, axis=0)
            )

            self.parameter_w -= self.learning_rate * (
                np.sum((y_hat.T - y.T) * self.parameter_v * Z * ( 1- Z) * X, axis=0)
            )
