#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import pandas as pd
np.set_printoptions(threshold=np.inf)


class DataLengthNotMatchError(Exception):

    def __init__(self):
        super(DataLengthNotMatchError, self).__init__(
            "The length of predicted data and testing data is not matching"
        )


class GaussianDiscriminantAnalysis(object):

    def __init__(self):
        self.mean_ = None
        self.variance_ = None

    def compute_parameters(self, training_data, class_col_idx):
        # The type of data shoudl be pandas's Dataframe.
        # we use this datatype fot the sake of better filter data based
        # on specific value
        if not isinstance(training_data, pd.core.frame.DataFrame):
            raise TypeError("DataFrame instance needed")
        self.class_col_idx = class_col_idx
        self.class_column_name = training_data.columns[class_col_idx]
        y = training_data.as_matrix()[:, class_col_idx]
        self.n_sample = y.shape[0]
        self._split_class(y)

    def _split_class(self, y):
        from collections import Counter
        # Compute of the number each class apppearing in training data set using
        # buildin python tool
        class_count = Counter(y)
        self._class_list = [k for k in class_count.keys()]

    def train(self, training_data, class_col_idx):
        self.compute_parameters(training_data, class_col_idx=class_col_idx)

    def prior(self, class_):
        # Compute the possibility of p(y=classj)
        count = 0
        for sample_index in range(self.n_sample):
            try:
                if self._y[sample_index] == class_:
                    count += 1
            except:
                # Ingore invalid value
                pass
        # print "class %s %s" % (class_, str(count))
        return count/self.n_sample

    def is_parameter_valid(self, predicted_class, testing_class):
        if predicted_class.shape[0] != testing_class.shape[0]:
            return False
        return True

    def predict(self, testing_sample):
        # transform datatype into 'float' for computation
        # testing_sample = np.array(testing_sample, dtype=np.float)
        testing_sample = testing_sample.as_matrix()[:, :-1]
        n_sample = testing_sample.shape[0]

        predicted_class = []
        for sample_index in range(n_sample):
            maximum_likelihood = -np.inf
            predicted = ""
            for c in self._class_list:
                pro = self.likelihood(testing_sample[sample_index, :], c)
                print "pro= %.2f" %pro
                if pro > maximum_likelihood:
                    maximum_likelihood = pro
                    predicted = c
            predicted_class.append(predicted)
        # print predicted_class
        return np.array(predicted_class)

    def confusion_matrix(self, predicted_class, testing_class, selected_class=None):
        # selected_class defined as the value we decided as positive
        if not self.is_parameter_valid(predicted_class, testing_class):
            raise DataLengthNotMatchError
        # Compute each variable as 0
        if not selected_class:
            selected_class = self._class_list[0]
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        n_sample = predicted_class.shape[0]
        for sample_index in range(n_sample):
            if testing_class[sample_index] == selected_class:
                if predicted_class[sample_index] == testing_class[sample_index]:
                    true_positive += 1
                else:
                    false_positive += 1
            else:
                if predicted_class[sample_index] == testing_class[sample_index]:
                    true_negative += 1
                else:
                    false_negative += 1
        print "TP %d" %true_positive
        print "FP %d" %false_positive
        print "TN %d" %true_negative
        print "FN %d" %false_negative

        if (true_positive + false_positive) == 0:
            self.precision = np.nan
        else:
            self.precision = (true_positive) / (true_positive + false_positive)
        if (true_positive + false_positive) == 0:
            self.recall = np.nan
        else:
            self.recall = true_negative / (true_negative + false_negative)
        if self.precision == np.nan or self.recall == np.nan:
            self.f_measure = np.nan
        else:
            self.f_measure = (2*self.recall * self.precision / \
                                    (self.precision + self.recall))

        self.accuracy = (true_positive + true_negative) / n_sample
        self.precision = true_positive /(true_positive + false_positive)
        self.recall = true_positive / (true_positive + false_negative)
        self.f_measure= 2*self.precision * self.recall/ (self.precision + self.recall)
        print "Accuracy = %.2f" % self.accuracy
        print "Precision = %.2f" % self.precision
        print "Recall = %.2f" % self.recall
        print "F-Measure = %.2f" % self.f_measure

    # def perform(self):
        # if not self.accuracy:
            # print "Please compute confusion matrix first"
            # return
        # print "Accuracy = %.2f" % self.accuracy
        # print "Precision = %.2f" % self.precision
        # print "Recall = %.2f" % self.recall
        # print "F-measure = %.2f" % self.f_measure


class SingleDimensionTwoClassGDA(GaussianDiscriminantAnalysis):

    def __init__(self):
        super(SingleDimensionTwoClassGDA, self).__init__()

    def compute_parameters(self, training_data, class_col_idx):
        super(SingleDimensionTwoClassGDA, self).compute_parameters(training_data, class_col_idx)
        self._compute_main_and_variance(training_data)

    def _compute_main_and_variance(self, training_data):
        # If axis=0, then this function will compute the mean of column.
        self.mean_ = []
        self.variance_ = []
        for idx, c in enumerate(self._class_list):
            class_column_name = training_data.columns[self.class_col_idx]
            self.mean_.append(
                training_data[training_data[class_column_name] == c].mean()[0]
            )
            self.variance_.append(
                training_data[training_data[class_column_name] == c].var()[0]
            )

    def likelihood(self, x, class_):
        class_idx = self._class_list.index(class_)
        log_likelihood = np.log(1/np.sqrt(2 * np.pi)) -\
            np.log(self.variance_[class_idx]) +\
            ((-1/2) * ((x - self.mean_[class_idx])/self.variance_[class_idx]) ** 2) +\
            self.prior(class_)
        return log_likelihood


class MultiDimensionsTwoClassGDA(GaussianDiscriminantAnalysis):

    def __init__(self):
        super(MultiDimensionsTwoClassGDA, self).__init__()

    def compute_parameters(self, training_data, class_col_idx):
        super(MultiDimensionsTwoClassGDA, self).compute_parameters(training_data, class_col_idx)
        self._compute_main_and_covariance_matrix(training_data)
        self._dimension = training_data.shape[1] - 1

    def _compute_main_and_covariance_matrix(self, training_data):
        self.mean_ = training_data.groupby(self.class_column_name).mean().as_matrix()
        self.cov_ = []
        grouped_cov = training_data.groupby(self.class_column_name).cov()
        for c in self._class_list:
            self.cov_.append(grouped_cov.loc[c].as_matrix())

    def likelihood(self, x, class_):
        class_idx = self._class_list.index(class_)
        log_likelihood = -(2/self._dimension * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(self.cov_[class_idx]))) - \
            0.5 * np.dot(
                np.dot(
                    (x - self.mean_[class_idx]).T, np.linalg.inv(self.cov_[class_idx])),
                    (x - self.mean_[class_idx])
            ) + self.prior(class_)
        return log_likelihood
