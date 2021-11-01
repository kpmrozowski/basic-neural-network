import numpy as np
# tests:
file_sizes = np.array([100, 500, 1000, 10000])
learning_rates = np.array([ 2 ** i for i in range(-4,6) ])
relaxation = np.array([ 0.05, 0.1, 0.2, 0.4, 0.8 ])
number_of_hidden_layers = np.array([ 0, 1, 2, 3, 4 ])
number_of_neurons_in_layer = np.array([ 0, 1, 2, 3, 4 ])
classification_file_types = [ 'simple', 'three_gauss' ]
regression_file_types = [ 'activation', 'cube' ]
classification_cost_funcions = ['cross_entropy', '?']
classification_cost_funcions = ['pseudo_huber_loss', 'root_mean_square_error']

