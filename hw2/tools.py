#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np


def cross_validation(data, testing_data_index, folder=10):
    '''
    Split date into training data and testing data two parts by 'folder' prameter,
    default value of folder is 10
    '''
    length = data.shape[0]
    data_partitions = np.array_split(data, folder)
    testing_data = data_partitions[testing_data_index]
    testing_data_size = testing_data.shape[0]
    del data_partitions[testing_data_index]
    training_data = np.concatenate([seq for seq in data_partitions])
    training_data_size = training_data.shape[0]

    # Obtain random index sequence by given numpy random function
    # Divide data by 'folder'

    return testing_data, testing_data_size, training_data, training_data_size
