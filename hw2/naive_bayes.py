#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import pandas as pd


class BasicNaiveBayes(object):

    def __init__(self):
        pass

    def _split_class_and_compute_parmeters(self, y):
        from collections import Counter
        # Compute of the number each class apppearing in training data set using
        # buildin python tool
        self.n_sample = len(y)
        class_count = Counter(y)
        self._class_list = [k for k in class_count.keys()]
        self.priors = {}
        for label in self._class_list:
            self.priors[label] = [class_count[label]/self.n_sample]
        self.priors = pd.DataFrame(self.priors)

class TwoDimensionNB(BasicNaiveBayes):

    def __init__(self):
        super(TwoDimensionNB, self).__init__()

    def _compute_probability_of_each_word(self, training_data, word, class_):
        return training_data[(training_data[word] == 1) & (training_data["class"] == class_)].shape[0]/self.n_sample

    def train(self, training_data):
        self.words_vector = training_data.columns[:-1]
        self._split_class_and_compute_parmeters(training_data.as_matrix()[:, -1])
        self.probability_table = []
        self.probability_vector = {}
        for class_ in self._class_list:
            for word in training_data.columns:
                self.probability_vector[word] = \
                    [self._compute_probability_of_each_word(
                        training_data,
                        word,
                        class_=class_
                    )]
            self.probability_table.append(pd.DataFrame(self.probability_vector))
        self.probability_table = pd.concat(self.probability_table, keys=self._class_list)

    def classify(self, test_word_vector):
        '''
        The data type for test_word_vector is pandas DataFrame:

            #  word1  word2  word3
            0    1      0      1

        '''
        max_likelihood = -np.inf
        predicted_class = None
        for c in self._class_list:
            likelihood = 0
            for each_word in self.words_vector:
                likelihood += self.log_bernoulli_function(
                    self.probability_table.ix[int(c)][each_word][0],
                    test_word_vector[each_word]
                )
            likelihood += self.priors[c][0]
            if likelihood > max_likelihood:
                max_likelihood = likelihood
                predicted_class = c
        return predicted_class

    def log_bernoulli_function(self, alpha, x):
        a = x * np.log(alpha) + (1 - x) * np.log(1 - alpha)
        return a

    def predict(self, test_data):
        n_test_sample = test_data.shape[0]
        predicted_list = []
        for idx in test_data.index:
            # print idx
            test_sample = test_data.ix[idx]
            predicted_list.append(
                self.classify(test_sample)
            )
        return predicted_list

    def perform(self, predicted, test):
        count = 0
        for i in range(len(predicted)):
            if predicted[i] == test[i]:
                count += 1
        print count


