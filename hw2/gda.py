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


class GaussianDistributionAnalysis(object):

    def __init__(self):
        self.mean_ = None
        self.variance_ = None

    def compute_parameters(self, training_data, class_col_idx):
        # The type of data shoudl be pandas's Dataframe.
        # we use this datatype fot the sake of better filter data based
        # on specific value
        if not isinstance(training_data, pd.core.frame.DataFrame):
            raise TypeError("DataFrame instance needed")
        self.class_column_name = training_data.columns[class_col_idx]
        y = training_data.as_matrix()[:, class_col_idx]
        self.n_sample = y.shape[0]
        self._split_class(y)
        self._compute_main_and_variance(training_data)

    def _split_class(self, y):
        from collections import Counter
        # Compute of the number each class apppearing in training data set using
        # buildin python tool
        class_count = Counter(y)
        self._class_list = [k for k in class_count.keys()]

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

    def predict(self, testing_data):
        # Initialize a new empty np array to store predicted value
        n_testing_sample = testing_data.shape[0]
        predicted_class = np.empty(n_testing_sample, 1)
        for sample_index in range(n_testing_sample):
            predicted_class[sample_index] = self.classify(
                testing_data[
                    sample_index,
                    :])
        return predicted_class


    def is_parameter_valid(self, predicted_class, testing_class):
        if predicted_class.shape[0] != testing_class.shape[0]:
            return False
        return True

class SingleDimensionTwoClassGDA(GaussianDistributionAnalysis):

    def __init__(self):
        super(SingleDimensionTwoClassGDA, self).__init__()

    def train(self, training_data, class_col_idx):
        self.compute_parameters(training_data, class_col_idx=class_col_idx)

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

    def classify(self, testing_sample):
        # transform datatype into 'float' for computation
        # testing_sample = np.array(testing_sample, dtype=np.float)
        testing_sample = testing_sample.as_matrix()[:, :-1]
        n_sample = testing_sample.shape[0]

        predicted_class = []
        for sample_index in range(n_sample):
            maximum_likelihood = -np.inf
            predicted = ""
            for c in self._class_list:
                p = self.likelihood(testing_sample[sample_index, :], c)
                if p > maximum_likelihood:
                    maximum_likelihood = p
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

        self.accuracy = (true_positive + true_negative) / n_sample
        self.precision = true_positive /(true_positive + false_positive)
        self.recall = true_positive / (true_positive + false_negative)
        self.f_measure= 2*self.precision * self.recall/ (self.precision + self.recall)

    def perform(self):
        if not self.accuracy:
            print "Please compute confusion matrix first"
            return
        print "Accuracy = %.2f" % self.accuracy
        print "Precision = %.2f" % self.precision
        print "Recall = %.2f" % self.recall
        print "F-measure = %.2f" % self.f_measure

class MultiDimensionsTwoClassGDA(GaussianDistributionAnalysis):

    def __init__(self):
        super(SingleDimensionTwoClassGDA, self).__init__()

    def train(self, training_data, class_col_idx):
        pass

