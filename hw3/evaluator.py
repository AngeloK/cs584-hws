#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import pandas as pd

class Evaluator(object):

    def __init__(self, predicted_condition, true_condition, positive, class_dim=2):
        self.predicted_condition = predicted_condition
        self.n_sample = predicted_condition.shape[0]
        self.true_condition = true_condition
        self.postive = positive
        self.class_dim = class_dim
        self.accuracy = None

    def score(self):
        if self.class_dim == 2:
            true_positive = 0
            true_negative = 0
            false_positive = 0
            false_negative = 0
            for idx in self.predicted_condition.index:
                if self.predicted_condition.ix[idx][0] == self.postive:
                    if self.predicted_condition.ix[idx][0] == \
                            self.true_condition.ix[idx][0]:
                        true_positive += 1
                    else:
                        false_positive += 1
                else:
                    if self.predicted_condition.ix[idx][0] == \
                            self.true_condition.ix[idx][0]:
                        true_negative += 1
                    else:
                        false_negative += 1
            print "TP = %d" % true_positive
            print "TN = %d" % true_negative
            print "FP = %d" % false_positive
            print "FN = %d" % false_negative
            if (true_positive + false_positive) == 0:
                self.precision = np.nan
            else:
                self.precision = (true_positive) / (true_positive + false_positive)
            if (true_positive + false_positive) == 0:
                self.recall = np.nan
            else:
                self.recall = true_positive / (true_positive + false_negative)
            if self.precision == np.nan or self.recall == np.nan:
                self.f_measure = np.nan
            else:
                self.f_measure = (2*self.recall * self.precision / \
                                        (self.precision + self.recall))
            self.accuracy = (true_positive + true_negative)/ self.n_sample
            print "Accuracy = %.2f" % self.accuracy
            print "Precision = %.2f" % self.precision
            print "Recall = %.2f" % self.recall
            print "F-Measure = %.2f" % self.f_measure
        else:
            correct = 0
            df_index = df_column = [i for i in range(self.class_dim)]
            for idx in self.predicted_condition.index:
                if self.predicted_condition.ix[idx][0] == self.true_condition.ix[idx][0]:
                    correct += 1
            self.accuracy = correct/self.n_sample
            result_table = np.zeros((self.class_dim, self.class_dim))
            for idx in self.true_condition.index:
                result_table[self.true_condition.ix[idx][0], self.predicted_condition.ix[idx][0]] += 1
            print "Accuracy = %.2f" % self.accuracy
            return pd.DataFrame(result_table, index=df_index, columns=df_column)

