from os import read
import numpy as np
import reader
import sklearn.preprocessing


def preprocess(xy, z):
    # xy, z is n*1 matrix
    # xy, z = reader.read_data()
    xy = sklearn.preprocessing.normalize(xy.T, norm="l2", axis=1)
    z = sklearn.preprocessing.normalize(z.T, norm="l2", axis=1)
    return xy, z
