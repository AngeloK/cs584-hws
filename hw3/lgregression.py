#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np


def sigmoid(matZ):
    return 1.0/(1+np.exp(-matZ))


def softmax(X, cls_idx, thetas):
    prior_sum = 0

    for i in range(len(thetas)):
        prior_sum += np.exp(np.dot(X, thetas[i]))

    return np.exp(np.dot(X, thetas[cls_idx])) / prior_sum


class LogisticRegression(object):

    def __init__(self):
        self.learning_rate = None

    def fit(self, X, y, learning_rate, iteration=100):
        self.learning_rate = learning_rate
        m, n = np.shape(X)
        thetas = np.ones((n, 1))

        # Gradient descent
        for i in range(iteration):
            h = sigmoid(np.dot(X, thetas))
            y = y.reshape((X.shape[0], 1))
            temp_thetas = self.learning_rate * (np.sum((h - y) * X, axis=0))
            temp_thetas = temp_thetas.reshape(thetas.shape)
            thetas = thetas - temp_thetas
        self.coef_ = thetas

    def predict(self, X_predict):
        y_predict = np.dot(X_predict, self.coef_)
        y_label = []
        for i in y_predict:
            if i[0] > 0:
                y_label.append(1)
            else:
                y_label.append(0)
        return np.array(y_label)


class KClassLogisticRegression(object):

    def __init__(self):
        self.learning_rate = None
        self.attr_count = 0
        self._class_list = None

    def fit(self, X, y, learning_rate, iteration=100):
        self.learning_rate = learning_rate
        m, n = np.shape(X)
        self.attr_count = n
        self._split_class(y)
        thetas = []

        # Initialize paramters.
        for i in range(len(self._class_list)):
            theta = np.ones((n, 1))
            thetas.append(theta)

        for i in range(iteration):
            for idx, c in enumerate(self._class_list):
                h = softmax(X, idx, thetas)
                indicator = self.indicator(y, c)
                indicator = indicator.reshape((X.shape[0], 1))
                temp_theta = self.learning_rate * (np.sum((h-indicator) * X, axis=0))
                temp_theta = temp_theta.reshape(theta.shape)
                thetas[idx] = thetas[idx] - temp_theta
        self.coef_ = thetas

    def _split_class(self, y):
        from collections import Counter
        # Using build-in module to compute the count of each class.
        class_count = Counter(y)
        self._class_list = [k for k in class_count.keys()]

    def indicator(self, y, label):
        new_y = []
        for l in y:
            if l == label:
                new_y.append(1)
            else:
                new_y.append(0)
        return np.array(new_y)

    def predict(self, X_test):
        y_predict = []
        for test_sample in X_test:
            likelihood = -np.inf
            target = None
            # test_sample = test_sample.reshape((1, self.attr_count))
            for idx, parameter in enumerate(self.coef_):
                l = np.dot(parameter.T, test_sample)
                if l > likelihood:
                    likelihood = l
                    target = idx
            y_predict.append(target)
        return np.array(y_predict)



