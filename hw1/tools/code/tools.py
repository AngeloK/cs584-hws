#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def read_data(filename):
    data = []
    with open(filename) as f:
        for line_index, line in enumerate(f):
            if line_index >= 5:
                data.append(line.strip().split(" "))
    np_data = np.array(data, dtype="f")
    return np_data


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


def data_split(data, folder=10):
    '''
    Split date into training data and testing data two parts by 'folder' prameter,
    default value of folder is 10
    '''
    length = data.shape[0]

    # Obtain random index sequence by given numpy random function
    random_seq = np.random.permutation(length)

    # Divide data by 'folder'
    test_data_size = length//10
    training_data_size = length - test_data_size
    test_data = data[:test_data_size]
    training_data = data[test_data_size+1:]

    return test_data, test_data_size, training_data, training_data_size


def plot_error(error_list, training_error_list):
    number_of_plot = len(error_list)
    plot_num = len(error_list)
    plot_index = 1
    for p_index in range(plot_num):
        plt.subplot(number_of_plot/2, 2, plot_index)
        plt.title("Dataset %s" %str(plot_index))
        plt.xlabel("Degree")
        plt.ylabel("Testing/Training Error")
        e = error_list[p_index]
        t = training_error_list [p_index]
        x_val = [data[0] for data in e.items()]
        y_val = [data[1] for data in e.items()]
        x_training_val = [data[0] for data in t.items()]
        y_training_val = [data[1] for data in t.items()]
        plt.plot(x_val, y_val, color="green", label="Testing Error")
        plt.plot(x_training_val, y_training_val, color="red", label="Training Error")
        plt.legend(loc='upper right')
        plot_index += 1
    plt.show()


def plot_time_cost(time_collection):
    number_of_plot = len(time_collection)
    plot_index = 1
    for t in time_collection:
        x_val = [data[0] for data in t.items()]
        y_val = [data[1] for data in t.items()]
        plt.subplot(number_of_plot/2, 2, plot_index)
        plt.title("Dataset %s" %str(plot_index))
        plt.xlabel("Degree")
        plt.ylabel("Time")
        plt.plot(x_val, y_val, color="green", label="Testing Error")
        plot_index += 1
    plt.show()
