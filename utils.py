# import pandas as pd
import numpy as np
from matplotlib import pyplot, colors

def read_csv_file(method, file, problem, count):
    # AttributeError: 'str' object has no attribute 'astype'
        # method = method.astype(str)
        # file = file.astype(str)
        # problem = type.astype(str)
        # count = count.astype(str)

    # csv_file = pd.read_csv("data/{}/data.{}.{}.{}.csv".format(method, file, problem, count))
    if file == 'mnist' and type == 'train':
        return np.genfromtxt('out/mnist_784.csv', delimiter=',', skip_header = 1, usecols = range(0,785))
    elif file != 'mnist':
        return np.genfromtxt("data/{}/data.{}.{}.{}.csv".format(method, file, problem, count), delimiter=',', skip_header = 1) # usecols = range(0,3))
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

def plot_set(set):
    set_coords = set[:,:-1]
    cls = set[:,-1].astype(int)

    min_cls = np.min(cls)
    max_cls = np.max(cls)
    classes_count = max_cls - min_cls + 1
    
    col = ['red','green','blue']
    fig = pyplot.figure(figsize=(8,8))
    pyplot.scatter(set_coords[:,0], set_coords[:,1], c=cls, cmap=colors.ListedColormap(col[:classes_count]))
    cb = pyplot.colorbar()
    loc = np.arange(0,max(cls),max(cls)/float(classes_count))
    cb.set_ticks(loc)
    cb.set_ticklabels(col[:classes_count])
    pyplot.show()

def plot_classification(set, predictions):
    set_coords = set[:,:-1]
    cls = set[:,-1].astype(int)

    min_cls = np.min(cls)
    max_cls = np.max(cls)
    classes_count = max_cls - min_cls + 1
    
    col = ['red','green','blue']
    fig, (ax0, ax1) = pyplot.subplots(1,2, figsize=(8,8))
    fig.suptitle('Final predictions!')
    l00 = ax0.scatter(set_coords[:,0], set_coords[:,1], marker="o",  s=.5, c=cls, cmap=colors.ListedColormap(col[:classes_count]))
    l10 = ax1.scatter(set_coords[:,0], set_coords[:,1], marker="o",  s=.5, c=predictions, cmap=colors.ListedColormap(col[:classes_count]))
    ax0.set_title('Testining set')
    ax1.set_title('predictions')
    ax0.set(xlabel='x', ylabel='y')
    ax1.set(xlabel='x', ylabel='y')
    # pyplot.legend((l00), ('Y_train', 'predictions'), loc='upper right', shadow=True)
    # pyplot.legend((l10), ('Y_test', 'predictions'), loc='upper right', shadow=True)
    # cb = pyplot.colorbar()
    # loc = np.arange(0,max(cls),max(cls)/float(classes_count))
    # cb.set_ticks(loc)
    # cb.set_ticklabels(col[:classes_count])
    # pyplot.show()
    pyplot.savefig('classification-predictions.png', dpi=300)