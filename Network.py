import json
import argparse
import time
from posix import XATTR_SIZE_MAX
import numpy as np
from matplotlib import pyplot, colors
from mnist import Networkm
from utils import read_csv_file, comp_confmat #, DrawNN

class Network:
    def __init__(self, training_inputs=None, training_outputs=None, cost_function='pseudo_huber', activation_function='sigmoid', problem='classification'):
        self.layers_count = 0
        self.layers = []
        self.problem = problem
        self.X_raw = np.array(training_inputs)
        self.Y_raw = np.array(training_outputs)
        self.X_train = np.array([])
        self.Y_train = np.array([])
        self.X_verif = np.array([])
        self.Y_verif = np.array([])
        self.cost_function = cost_function
        self.activation_function = activation_function
        self.huber_delta = 1.
        self.bounds_raw = np.zeros((4))
    
    def add_first_hidden(self, neurons_count, inputs_count, biases = None, weights_mat = None):
        input_layer = Layer(inputs_count, None, None)
        self.layers.append(Layer(neurons_count, biases, weights_mat, input_layer, self.activation_function))
        self.layers_count += 1
        self.num_inputs = inputs_count
        self.output_layer = self.layers[self.layers_count - 1]

    def add(self, neurons_count, biases = None, weights_mat = None):
        self.layers.append(Layer(neurons_count, biases, weights_mat, \
                           self.layers[self.layers_count - 1], self.activation_function))
        self.layers_count += 1
        self.output_layer = self.layers[self.layers_count - 1]

    def print_myself(self):
        for layer in self.layers:
            layer.update()
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        for i in range(len(self.layers) - 1):
            print('------')
            print('Hidden Layer')
            self.layers[i].print_myself()
        print('------')
        print('* Output Layer')
        self.layers[-1].print_myself()
        print('------')
    
    def draw_myself(self):
        for layer in self.layers:
            layer.draw_myself()
        pyplot.axis('scaled')
        pyplot.show()

    def inference(self, inputs):
        self.outputs = inputs
        for layer in self.layers:
            self.outputs = layer.inference(inputs)
        result = self.output_layer.inference(self.outputs)
        return result

    def feed_forward(self, X):
        self.layers[0].compute_outputs(X)
        for i in range(1, len(self.layers)-1):
            self.layers[i].compute_outputs(self.layers[i-1].axons_outputs)
        if self.problem == 'classification':
            if len(self.layers) == 1:
                self.layers[-1].soft_max(X)
            else:
                self.layers[-1].soft_max(self.layers[-2].axons_outputs)
        elif self.problem == 'regression':
            if len(self.layers) == 1:
                self.layers[-1].sum_of_inputs(X)
            else:
                self.layers[-1].sum_of_inputs(self.layers[-2].axons_outputs)
    
    def predict(self, X_raw):
        if self.problem == 'classification':
            self.feed_forward(X_raw.T)
            predictions = np.argmax(self.layers[-1].axons_outputs.T, axis=1)
            Y_raw = predictions
        elif self.problem == 'regression':
            X = np.divide( X_raw - self.bounds_raw[0], self.bounds_raw[1] - self.bounds_raw[0] )
            self.feed_forward(np.array([X]))
            Y = self.layers[-1].axons_outputs
            Y_raw = self.bounds_raw[2] + np.multiply( Y, self.bounds_raw[3] - self.bounds_raw[2] )
        return Y_raw

    def back_propagate(self, X, Y, m_batch):
        # derivatives: https://stats.stackexchange.com/questions/370723/how-to-calculate-the-derivative-of-crossentropy-error-function
        a = self.layers[-1].axons_outputs - Y
        if self.cost_function == 'pseudo_huber':
            dZ = np.square( self.huber_delta ) * a / np.sqrt(1 + np.square( np.divide( a, self.huber_delta ) ) )
        elif self.cost_function == 'mean_squared_error':
            dZ = a
        elif self.cost_function == 'cross_entropy':
            Y_predicted = self.layers[-1].axons_outputs
            Y_predicted[ Y_predicted > 1 - 1e-6 ] = 1 - 1e-6
            Y_predicted[ Y_predicted < 1e-6 ] = 1e-6
            dZ = - np.divide( Y, Y_predicted ) + np.divide( 1 - Y, 1 - Y_predicted )
        if len(self.layers) == 1:
            self.layers[-1].dW = (1./m_batch) * np.matmul(dZ, X.T)
        else:
            self.layers[-1].dW = (1./m_batch) * np.matmul(dZ, self.layers[-2].axons_outputs.T)
        self.layers[-1].db = (1./m_batch) * np.sum(dZ, axis=1, keepdims=True)
        for i in range(len(self.layers)-2, 0, -1):
            dA = np.matmul(self.layers[i+1].weights.T, dZ)
            A = self.layers[i].axons_outputs
            if self.activation_function == 'sigmoid':
                dZ = dA * A * (1 - A)
            if self.activation_function == 'relu':
                dZ = dA * (A > 0) * 1
            if self.activation_function == 'tanh':
                dZ = dA * (1 - A ** 2)
            self.layers[i].dW = (1./m_batch) * np.matmul(dZ, self.layers[i-1].axons_outputs.T)
            self.layers[i].db = (1./m_batch) * np.sum(dZ, axis=1, keepdims=True)
        if len(self.layers) > 1:
            dA = np.matmul(self.layers[1].weights.T, dZ)
            A = self.layers[0].axons_outputs
            if self.activation_function == 'sigmoid':
                dZ = dA * A * (1 - A)
            if self.activation_function == 'relu':
                dZ = dA * (A > 0) * 1
            if self.activation_function == 'tanh':
                dZ = dA * (1 - A ** 2)
        self.layers[0].dW = (1./m_batch) * np.matmul(dZ, X.T)
        self.layers[0].db = (1./m_batch) * np.sum(dZ, axis=1, keepdims=True)
    
    def train(self, batch_size, train_verif_ratio, learning_rate, relaxation, timeout, iterations_initial=100, remote=0, huber_delta=1.):
        statistics = {
            "epoch": [],
            "train_cost": [],
            "verif_cost": [],
            "accuracy_train": [],
            "accuracy_verif": [] }
        m = int((self.X_raw.shape[0] * train_verif_ratio))
        if self.problem == 'classification':
            min_cls = np.min(self.Y_raw)
            max_cls = np.max(self.Y_raw)
            classes_count = max_cls - min_cls + 1
            Y = np.eye(classes_count)[self.Y_raw - min_cls]
            self.X_train, self.X_verif = self.X_raw[:m].T, self.X_raw[m:].T
            self.Y_train, self.Y_verif = Y[:m].T, Y[m:].T
        elif self.problem == 'regression':
            self.bounds_raw = np.array([self.X_raw.min(), self.X_raw.max(), self.Y_raw.min(), self.Y_raw.max()])
            Y = np.divide( self.Y_raw - self.bounds_raw[2], self.bounds_raw[3] - self.bounds_raw[2] )
            X = np.divide( self.X_raw - self.bounds_raw[0], self.bounds_raw[1] - self.bounds_raw[0] )
            self.X_train, self.X_verif = np.array([X[:m]]), np.array([X[m:]])
            self.Y_train, self.Y_verif = np.array([Y[:m]]), np.array([Y[m:]])
            self.huber_delta = huber_delta
        batches = -(-m // batch_size)
        epoch = 0
        iterations_left = iterations_initial
        while True:
            permutation = np.random.permutation(m)
            X_train_shuffled = self.X_train[:, permutation]
            Y_train_shuffled = self.Y_train[:, permutation]
            for i in range(batches):
                begin = i * batch_size
                end = min(begin + batch_size, m - 1)
                X = X_train_shuffled[:, begin:end]
                Y = Y_train_shuffled[:, begin:end]
                m_batch = end - begin
                self.feed_forward(X)
                self.back_propagate(X, Y, m_batch)

                for layer in self.layers:
                    layer.V_dW = (1. - relaxation) * layer.V_dW + relaxation * layer.dW
                    layer.V_db = (1. - relaxation) * layer.V_db + relaxation * layer.db
                    layer.weights -= learning_rate * layer.V_dW
                    layer.biases -= learning_rate * layer.V_db
            if self.cost_function == 'cross_entropy':
                train_cost = self.cross_entropy_loss(self.X_train, self.Y_train)
                verif_cost = self.cross_entropy_loss(self.X_verif, self.Y_verif)
            elif self.cost_function == 'pseudo_huber':
                train_cost = self.pseudo_huber_loss(self.X_train, self.Y_train)
                verif_cost = self.pseudo_huber_loss(self.X_verif, self.Y_verif)
            elif self.cost_function == 'mean_squared_error':
                train_cost = self.mean_squared_error(self.X_train, self.Y_train)
                verif_cost = self.mean_squared_error(self.X_verif, self.Y_verif)
            E_train = self.distance_error(self.X_train, self.Y_train)
            E_verif = self.distance_error(self.X_verif, self.Y_verif)
            accuracy_train = 100 * np.divide( np.sum( E_train < .001 ), E_train.shape[1] )
            accuracy_verif = 100 * np.divide( np.sum( E_verif < .001 ), E_verif.shape[1] )
            statistics["epoch"].append(epoch)
            statistics["train_cost"].append(train_cost)
            statistics["verif_cost"].append(verif_cost)
            statistics["accuracy_train"].append(accuracy_train)
            statistics["accuracy_verif"].append(accuracy_verif)
            if epoch % 10 == 0 and remote == 0:
                print("Epoch:{},trainCost={:.4f}‰,verifCost={:.4f}‰".format(epoch+1, 1000*train_cost, 1000*verif_cost))
            if iterations_left < 0:
                if self.problem == 'classification' and remote == 0:
                    predictions = np.argmax(self.layers[-1].axons_outputs.T, axis=1)
                    labels = np.argmax(self.Y_verif, axis=0)
                    print(comp_confmat(labels, predictions))
                    print('accuracy(verif) =', np.mean(predictions == labels))
                elif self.problem == 'regression' and remote == 0:
                    fig, (ax0, ax1) = pyplot.subplots(1,2)
                    fig.suptitle('Vertically stacked subplots')
                    self.feed_forward(self.X_verif)
                    Y = self.layers[-1].axons_outputs
                    E = np.abs( Y - self.Y_verif )
                    mse_verif = self.mean_squared_error(self.X_train, self.Y_train)
                    print('Mean Squared Error(verif)={:.2f}‰'.format(1000*mse_verif))
                    print("Accuracy (|Y-Yhat|<{}): {:.2f}%". format(.001, 100 * np.divide( np.sum( E < .001 ), Y.shape[1] ) ) )
                    ax0.set_title('Veryficating set, MSE={:.2f}‰'.format(1000*mse_verif))
                    X_verif = self.bounds_raw[0] + np.multiply( self.X_verif, self.bounds_raw[1] - self.bounds_raw[0] )
                    Y_verif = self.bounds_raw[2] + np.multiply( self.Y_verif, self.bounds_raw[3] - self.bounds_raw[2] )
                    Y_verif_pred = self.bounds_raw[2] + np.multiply( Y, self.bounds_raw[3] - self.bounds_raw[2] )
                    l00 = ax0.scatter(X_verif , Y_verif, marker="o",  s=2)
                    l01 = ax0.scatter(X_verif , Y_verif_pred, marker="o",  s=2)
                    ax0.legend((l00, l01), ('Y_verif', 'predictions'), loc='upper right', shadow=True)
                    self.feed_forward(self.X_train)
                    Y = self.layers[-1].axons_outputs
                    E = np.abs( Y - self.Y_train )
                    mse_train = self.mean_squared_error(self.X_verif, self.Y_verif)
                    print('Mean Squared Error(train)={:.2f}‰'.format(1000*mse_train))
                    print("Accuracy (|Y-Yhat|<{}): {:.2f}%". format(.001, 100 * np.divide( np.sum( E < .001 ), Y.shape[1] ) ) )
                    ax1.set_title('Training set, MSE={:.2f}‰'.format(1000*mse_train))
                    ax1.set(xlabel='x', ylabel='y')
                    X_train = self.bounds_raw[0] + np.multiply( self.X_train, self.bounds_raw[1] - self.bounds_raw[0] )
                    Y_train = self.bounds_raw[2] + np.multiply( self.Y_train, self.bounds_raw[3] - self.bounds_raw[2] )
                    Y_train_pred = self.bounds_raw[2] + np.multiply( Y, self.bounds_raw[3] - self.bounds_raw[2] )
                    l10 = ax1.scatter(X_train, Y_train, marker="o",  s=.5)
                    l11 = ax1.scatter(X_train, Y_train_pred, marker="o",  s=.5)
                    ax1.legend((l10, l11), ('Y_train', 'predictions'), loc='upper right', shadow=True)
                    pyplot.show()
                if remote == 0:
                    iterations_left_new = input("How many epochs do you want more?(default {}) Epochs: ".format(iterations_initial))
                    if iterations_left_new != '':
                        iterations_left += int(iterations_left_new)
                    elif len(iterations_left_new) > 0:
                        iterations_left += iterations_initial
                    if iterations_left == -1:
                        break
                    learning_rate_new = input("What learinig_rate do you want?(default {}) learning_rate=".format(learning_rate))
                    if learning_rate_new != '':
                        learning_rate = float(learning_rate_new)
                    huber_delta_new = input("What huber_delta do you want?(default {}) huber_delta=".format(self.huber_delta))
                    if huber_delta_new != '':
                        self.huber_delta = float(huber_delta_new)
                    relaxation_new = input("What relaxation parameter do you want?(default {}) relaxation=".format(relaxation))
                    if relaxation_new != '':
                        relaxation = float(relaxation_new)
                # if iterations_left < 0
            iterations_left -= 1
            epoch += 1
            if time.time() > timeout:
                break
            # while True
        for layer in self.layers:
            layer.update()

        self.feed_forward(self.X_train)
        Y = self.layers[-1].axons_outputs
        if np.sum( np.isnan(Y) ) + np.sum( np.isinf(Y) ) + np.sum( np.isneginf(Y) ) == 0:
            success = 1
        else:
            success = 0
        filename_header = '' \
            + "-a." + self.activation_function \
            + "-c." + self.cost_function \
            + "-d." + str(huber_delta) \
            + "-h." + str(self.layers_count) \
            + "-l." + str(learning_rate) \
            + "-n." + str(self.layers[0].neurons_count) \
            + "-p." + self.problem \
            + "-s." + str(success)
        if True:
            results_test = np.zeros([self.X_verif.T.shape[0], 3])
            results_test[:,:2] = self.X_verif.T
            self.feed_forward(self.X_verif)
            results_test[:,2] = np.argmax(self.layers[-1].axons_outputs.T, axis=1)
            np.savetxt('results_test' + filename_header + '.csv', results_test, delimiter=',')
            results_train = np.zeros([self.X_train.T.shape[0], 3])
            results_train[:,:2] = self.X_train.T
            self.feed_forward(self.X_train)
            results_train[:,2] = np.argmax(self.layers[-1].axons_outputs.T, axis=1)
            np.savetxt('results_train' + filename_header + '.csv', results_train, delimiter=',')
        if False:
            minX, maxX, minY, maxY = -1., 1., -1., 1.
            x = np.linspace(minX, maxX, 2**8)
            y = np.linspace(minY, maxY, 2**8)
            X, Y = np.meshgrid(x, y)
            X = X.reshape((np.prod(X.shape),))
            Y = Y.reshape((np.prod(Y.shape),))
            meshgrid = np.array([[X[i],Y[i]] for i in range(len(X))])

            results_meshgrid = np.zeros([meshgrid.shape[0], 3])
            results_meshgrid[:,:2] = meshgrid
            self.feed_forward(meshgrid.T)
            results_meshgrid[:,2] = np.argmax(self.layers[-1].axons_outputs.T, axis=1)
            np.savetxt("results_meshgrid_2classes.csv", results_meshgrid, delimiter=",")
        if True:
            results_statistics = np.array([statistics["epoch"], 
                                         statistics["train_cost"],
                                         statistics["verif_cost"], 
                                         statistics["accuracy_train"], 
                                         statistics["accuracy_verif"]]).T
            np.savetxt('results_statistics' + filename_header + '.csv', results_statistics, delimiter=",", header="epoch,train_cost,verif_cost,accuracy_train,accuracy_verif", comments='')
        if remote == 0:
            print("traininig done!")

    def error(self, training_sets, output_sets):
        error = 0
        for t in range(len(training_sets)):
            training_inputs = training_sets[t]
            training_outputs = output_sets[t]
            self.inference(training_inputs)
            for o in range(len(training_outputs)):
                error += self.layers[-1].neurons[o].error(training_outputs[o])
        return error

    def distance_error(self, X, Y_hat):
        self.feed_forward(X)
        Y = self.layers[-1].axons_outputs
        return np.abs( Y - Y_hat )

    # Loss functions: https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
    # L(Ŷ,Y)= −(1/m) * ∑_{i=0}^{m}{ ŷᵢ * log(yᵢ) + (1−ŷᵢ) * log(1−yᵢ) }.
    def cross_entropy_loss(self, X, Y_hat):
        self.feed_forward(X)
        m = Y_hat.shape[1]
        Y = self.layers[-1].axons_outputs
        Y[Y == 0.0] = 1e-6
        Y[Y == 1.0] = 1 - 1e-6
        a = np.multiply( np.log(Y), Y_hat)
        b = np.multiply( np.log(1-Y), 1-Y_hat)
        return - (1./m) * ( np.sum( a ) + np.sum( b ) )

    def huber_loss(self, X, Y_hat):
        self.feed_forward(X)
        Y = self.layers[-1].axons_outputs
        a = np.where(np.abs(Y-Y_hat) < self.huber_delta,
                     .5*(Y-Y_hat)**2 , 
                     self.huber_delta*(np.abs(Y-Y_hat)-0.5*self.huber_delta))
        return np.divide( np.sum( a ), Y_hat.shape[1] )

    # https://en.wikipedia.org/wiki/Huber_loss
    def pseudo_huber_loss(self, X, Y_hat):
        self.feed_forward(X)
        Y = self.layers[-1].axons_outputs
        return np.divide( np.sum( np.square( self.huber_delta ) * ( np.sqrt( 1 + np.square( np.divide( Y - Y_hat, self.huber_delta ) ) ) - 1 ) ) , Y_hat.shape[1] )

    def mean_squared_error(self, X, Y_hat):
        self.feed_forward(X)
        Y = self.layers[-1].axons_outputs
        return np.divide( np.sum( np.sqrt( np.mean( np.square(Y - Y_hat) ) ) ) , Y_hat.shape[1] )

class Layer:
    def __init__(self, neurons_count, biases, weights_mat, previous_layer = None, activation_function = 'sigmoid'):
        self.neurons_count = neurons_count
        self.biases = np.zeros((neurons_count, 1)) if biases is None else biases
        if weights_mat is None and previous_layer is not None:
            weights_mat = np.random.randn(neurons_count, previous_layer.neurons_count) * np.sqrt(1. / neurons_count)
        if previous_layer is not None:
            self.weights = weights_mat
            self.previous_layer = previous_layer
            self.activation_function = activation_function
            self.neurons = [ Neuron(weights_mat[i], self.biases[i]) for i in range(neurons_count) if self.biases is not None]
            self.synaps_inputs_sums = np.zeros([neurons_count, 1])
            self.axons_outputs = np.zeros([neurons_count, 1])
            self.dW = np.zeros([neurons_count, previous_layer.neurons_count])
            self.db = np.zeros([neurons_count, 1])
            self.V_dW = np.zeros([neurons_count, previous_layer.neurons_count])
            self.V_db = np.zeros([neurons_count, 1])

    def print_myself(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights.shape)):
                print('  Weight:', self.neurons[n].weights[w])
                print('  Bias:  ', self.biases[w])
    
    # def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons, number_of_neurons_in_widest_layer):
    #     horizontal_distance_between_neurons = 10
    #     # number_of_neurons_in_widest_layer = np.max(np.array([i for i in self.]))
    #     return horizontal_distance_between_neurons * (number_of_neurons_in_widest_layer - number_of_neurons) / 2

    # def __calculate_layer_y_position(self):
    #     if self.previous_layer:
    #         return self.previous_layer.y + vertical_distance_between_layers
    #     else:
    #         return 0

    def inference(self, inputs):
        self.inputs = inputs
        # yⱼ = σ(wⱼ.T * xⱼ + bⱼ)
        self.axons_outputs = np.array([neuron.axon_output(inputs) for neuron in self.neurons])
        return self.axons_outputs

    def compute_outputs(self, input_matrix):
        # y = σ(w.T * X + b)
        # print("shapes:", self.weights.shape, input_matrix.T.shape, self.biases.shape)
        self.synaps_inputs_sums = np.matmul(self.weights, input_matrix) + self.biases
        if self.activation_function == 'sigmoid':
            self.axons_outputs = self.sigmoid(self.synaps_inputs_sums)
        elif self.activation_function == 'relu':
            self.axons_outputs = self.relu(self.synaps_inputs_sums)
        elif self.activation_function == 'tanh':
            self.axons_outputs = self.tangensH(self.synaps_inputs_sums)
        else:
            print("No such activation_function:", self.activation_function)
            exit()
        return self.axons_outputs
    
    def soft_max(self, input_matrix):
        input_matrix[ np.isnan(input_matrix) ] = 0
        self.synaps_inputs_sums = np.matmul(self.weights, input_matrix) + self.biases
        sum = np.sum(np.exp(self.synaps_inputs_sums), axis=0)
        # sum[ np.abs(sum) < 1e-6 ] = 1e-6 * np.sign( sum[ np.abs(sum) < 1e-6 ] )

        self.axons_outputs = np.divide( np.exp(self.synaps_inputs_sums), sum )
        if np.sum( self.axons_outputs < 0 ) > 0:
            print('')
        return self.axons_outputs
    
    def sum_of_inputs(self, input_matrix):
        self.synaps_inputs_sums = np.matmul(self.weights, input_matrix) + self.biases
        self.axons_outputs = self.synaps_inputs_sums
        return self.axons_outputs
    
    def sigmoid(self, sum_of_inputs):
        return 1 / (1 + np.exp(-sum_of_inputs))
    def relu(self, sum_of_inputs):
        result = sum_of_inputs.copy()
        result[result < 0] = 0
        return result
    def tangensH(self, sum_of_input):
        return np.tanh(sum_of_input)
    
    def update(self):
        if self.neurons != []:
            for i in range(self.neurons_count):
                self.neurons[i].weights = self.weights[i]
                self.neurons[i].bias = self.biases[i]

class Neuron:
    def __init__(self, weights, bias = 0, activation_function = 'sigmoid'):
        self.weights = np.array(weights)
        self.activation_function = activation_function
        self.bias = bias

    def draw(self):
        circle = pyplot.Circle((self.x, self.y), radius=10, fill=False)
        pyplot.gca().add_patch(circle)
    # def axon_output(self, inputs):
    #     self.inputs = np.array(inputs)
    #     if self.activation_function == 'sigmoid':
    #         self.output = self.sigmoid(self.synaps_inputs_sum())
    #     elif self.activation_function == 'relu':
    #         self.output = self.relu(self.synaps_inputs_sum())
    #     elif self.activation_function == 'tanh':
    #         self.output = self.tangensH(self.synaps_inputs_sum())
    #     else:
    #         print("No such activation_function:", self.activation_function)
    #         exit()
    #     return self.output
    
    # def sigmoid(self, sum_of_inputs):
    #     return 1 / (1 + np.exp(-sum_of_inputs))
    # def relu(self, sum_of_inputs):
    #     return 0 if sum_of_inputs < 0 else sum_of_inputs
    # def tangensH(self, sum_of_input):
    #     return np.tanh(sum_of_input)

    # def synaps_inputs_sum(self):
    #     sum = 0
    #     for i in range(len(self.inputs)):
    #         sum += self.inputs[i] * self.weights[i]
    #     return sum + self.bias
    
    # # ∂E/∂zⱼ -->  the partial derivative of the neuron error with respect to the neuron input
    # # δ = ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ
    # def DsynapsError_DsynapsInputsSum(self, desired_output):
    #     delta = 0.
    #     if self.activation_function == 'sigmoid':
    #         delta = self.DsynapsError_Doutput(desired_output) * self.Doutput_DsynapsInputsSum_sigmoid()
    #     elif self.activation_function == 'relu':
    #         delta = self.DsynapsError_Doutput(desired_output) * self.Doutput_DsynapsInputsSum_relu()
    #     elif self.activation_function == 'tanh':
    #         delta = self.DsynapsError_Doutput(desired_output) * self.Doutput_DsynapsInputsSum_tanh()
    #     else:
    #         print("No such activation_function:", self.activation_function)
    #         exit()
    #     return delta

    # # The partial derivative of the neuron error with respect to actual neuron output:
    # # ∂E/∂yⱼ = 2 * 0.5 * (ŷⱼ - yⱼ) ^ (2 - 1) * -1 = 
    # #        = -(ŷⱼ - yⱼ) = -(desired output - actual output)
    # def DsynapsError_Doutput(self, desired_output):
    #     return -(desired_output - self.output)

    # # The derivative of the neuron output with respect to the total network input (sigmoid)
    # #      yⱼ = 1 / (1 + e^(-zⱼ))  | d/dzⱼ
    # # dyⱼ/dzⱼ = yⱼ * (1 - yⱼ)
    # def Doutput_DsynapsInputsSum_sigmoid(self):0.0000,verifCost=-
    # # The derivative of the neuron output with respect to the total network input (relu)
    # #      yⱼ = max(0, zⱼ)  | d/dzⱼ
    # # dyⱼ/dzⱼ = [1 if zⱼ > 0 else 0]
    # def Doutput_DsynapsInputsSum_relu(self):
    #     return 1 if self.output > 0 else 0

    # # The derivative of the neuron output with respect to the total network input (tanh)
    # #      yⱼ = tanh(zⱼ)  | d/dzⱼ
    # # dyⱼ/dzⱼ = 1 - (tanh(zⱼ))^2
    # # dyⱼ/dzⱼ = 1 - yⱼ^2
    # def Doutput_DsynapsInputsSum_tanh(self):
    #     return 1 - self.output ** 2

    # # Each neuron error is calculated like this:
    # # E = ŷⱼ - yⱼ
    # def error(self, desired_output):
    #     return 0.5 * (desired_output - self.output) ** 2

    # # The partial derivative of the total net input with respective to a given 
    # #     weight (with everything else held constant):
    # # ∂zⱼ/∂wᵢ = some constant + 1 * xᵢw₁^(1-0) + some constant ... = xᵢ
    # def DsynapsInputsSum_Dweight(self, index):
    #     return self.inputs[index]