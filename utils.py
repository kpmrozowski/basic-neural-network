# import pandas as pd
import numpy as np

def read_csv_file(method, file, type, count):
    # AttributeError: 'str' object has no attribute 'astype'
        # method = method.astype(str)
        # file = file.astype(str)
        # type = type.astype(str)
        # count = count.astype(str)

    # csv_file = pd.read_csv("data/{}/data.{}.{}.{}.csv".format(method, file, type, count))
    if file == 'mnist' and type == 'train':
        return np.genfromtxt('out/mnist_784.csv', delimiter=',', skip_header = 1, usecols = range(0,785))
    elif file != 'mnist':
        return np.genfromtxt("data/{}/data.{}.{}.{}.csv".format(method, file, type, count), delimiter=',', skip_header = 1) # usecols = range(0,3))
    return None

def comp_confmat(actual, predicted):

    # extract the different classes
    classes = np.unique(actual)

    # initialize the confusion matrix
    confmat = np.zeros((len(classes), len(classes)))

    # loop across the different combinations of actual / predicted classes
    for i in range(len(classes)):
        for j in range(len(classes)):

           # count the number of instances in each combination of actual / predicted classes
           confmat[i, j] = np.sum((actual == classes[i]) & (predicted == classes[j]))

    return np.array_str(confmat, suppress_small=True)