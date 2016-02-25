#!/usr/bin/env python
# -*- coding: utf-8 -*-


from gda import SingleVariateGDA
import numpy as np
import pandas as pd


if __name__ == "__main__":

    base_path = "/Users/Neyanbhbin/Documents/code/Data Analysis/cs584-hw2/2-class-dataset/bank/"
    sv = SingleVariateGDA()
    data = pd.read_csv(base_path+"2-d-bank.csv")
    print data
