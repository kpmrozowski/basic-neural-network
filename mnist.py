import json
import argparse
from posix import XATTR_SIZE_MAX
import numpy as np

from utils import read_csv_file, comp_confmat
from sklearn.metrics import classification_report

class Networkm:
    def __init__(self, training_inputs=None, training_outputs=None, activation_function='sigmoid', problem='classification'):
        self.layers_count = 0
        self.layers = []
        self.problem = problem
        self.X_raw = training_inputs
        self.Y_raw = training_outputs
        self.X_train = np.array([])
        self.Y_train = np.array([])
        self.X_verif = np.array([])
        self.Y_verif = np.array([])
        self.activation_function = activation_function

    def add_first_hidden(self, neurons_count, inputs_count, biases = None, weights_mat = None):
        self.layers.append(Layer(neurons_count, biases, weights_mat, inputs_count, self.activation_function))
        self.layers_count += 1
        self.num_inputs = inputs_count
        self.output_layer = self.layers[self.layers_count - 1]

    def add(self, neurons_count, biases = None, weights_mat = None):
        self.layers.append(Layer(neurons_count, biases, weights_mat, \
                           self.layers[self.layers_count - 1].neurons_count, self.activation_function))
        self.layers_count += 1
        self.output_layer = self.layers[self.layers_count - 1]

    def print_myself(self):
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

    def inference(self, inputs):
        self.outputs = inputs
        for layer in self.layers:
            self.outputs = layer.inference(inputs)
        result = self.output_layer.inference(self.outputs)
        # print('f({})={}'.format(inputs, result))
        return result

    def feed_forward(self, X):
        self.layers[0].compute_outputs(X)
        for i in range(1, len(self.layers)-1):
            self.layers[i].compute_outputs(self.layers[i-1].axons_outputs)
        self.layers[-1].soft_max(self.layers[-2].axons_outputs)

    def back_propagate(self, X, Y, m_batch):
        dZ = self.layers[-1].axons_outputs - Y
        self.layers[-1].dW = (1./m_batch) * np.matmul(dZ, self.layers[-2].axons_outputs.T)
        self.layers[-1].db = (1./m_batch) * np.sum(dZ, axis=1, keepdims=True)
        # print(dZ.shape)
        for i in range(len(self.layers)-2, 0, -1):
            dA = np.matmul(self.layers[i+1].weights.T, dZ)
            A = self.layers[i].axons_outputs
            dZ = dA * A * (1 - A)
            self.layers[i].dW = (1./m_batch) * np.matmul(dZ, self.layers[i-1].axons_outputs.T)
            self.layers[i].db = (1./m_batch) * np.sum(dZ, axis=1, keepdims=True)
            # print("shapes_57:", "W[1]", self.layers[i+1].weights.shape, "dA[0]", dA.shape, "A[0]", A.shape, "A[-1]", self.layers[i-1].axons_outputs.shape, "dW[0]", self.layers[i].dW.shape, "db", self.layers[i].db.shape)
        dA = np.matmul(self.layers[1].weights.T, dZ)
        A = self.layers[0].axons_outputs
        dZ = dA * A * (1 - A)
        self.layers[0].dW = (1./m_batch) * np.matmul(dZ, X.T)
        self.layers[0].db = (1./m_batch) * np.sum(dZ, axis=1, keepdims=True)
    
    def train(self, batch_size, train_verif_ratio, learning_rate, relaxation):
        m = int((self.X_raw.shape[0] * train_verif_ratio))
        if self.problem == 'classification':
            min_cls = np.min(self.Y_raw)
            max_cls = np.max(self.Y_raw)
            classes_count = max_cls - min_cls + 1
            Y = np.eye(classes_count)[self.Y_raw - min_cls]
        self.X_train, self.X_verif = self.X_raw[:m].T, self.X_raw[m:].T
        self.Y_train, self.Y_verif = Y[:m].T, Y[m:].T
        batches = -(-m // batch_size)
        epoch = 0
        iterations_left = 9
        while True:
            permutation = np.random.permutation(self.X_train.shape[1])
            X_train_shuffled = self.X_train[:, permutation]
            Y_train_shuffled = self.Y_train[:, permutation]
            for i in range(batches):
                begin = i * batch_size
                end = min(begin + batch_size, self.X_train.shape[1] - 1)
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

            self.feed_forward(self.X_train)
            train_cost = self.cross_entropy_loss()
            self.feed_forward(self.X_verif)
            verif_cost = self.cross_entropy_loss(self.Y_verif)
            print("Epoch {}: training cost = {}, verif cost = {}".format(epoch+1 ,train_cost, verif_cost))
            if iterations_left == 0:
                predictions = np.argmax(self.layers[-1].axons_outputs.T, axis=1)
                labels = np.argmax(self.Y_verif, axis=0)
                print(classification_report(predictions, labels))
                # print(confusion_matrix(predictions, labels))
                print(comp_confmat(labels, predictions))
                iterations_left_new = input("How many epochs do you want more?(default {}) Epochs: ".format(learning_rate))
                if iterations_left_new != '':
                    iterations_left += int(iterations_left_new)
                if iterations_left == 0:
                    break
                learning_rate_new = input("What learinig_rate do you want?(default {}) learning_rate=".format(learning_rate))
                if learning_rate_new != '':
                    learning_rate = float(learning_rate_new)
                relaxation_new = input("What relaxation parameter do you want?(default {}) relaxation=".format(relaxation))
                if relaxation_new != '':
                    relaxation = float(relaxation_new)
            iterations_left -= 1
            epoch += 1

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

    # L(Y??,Y)= ???(1/m) * ???_{i=0}^{m}{ y????? * log(y???) + (1???y?????) * log(1???y???) }.
    def total_error(self):
        error = 0
        for i in range(self.X_train.shape[0]):
            x = self.X_train[i]
            y = self.Y_train[i]
            self.inference(x)
            for o in range(len(y)):
                error += self.layers[-1].neurons[o].error(y[o])
        return error

    # L(Y??,Y)= ???(1/m) * ???_{i=0}^{m}{ y????? * log(y???) + (1???y?????) * log(1???y???) }.
    def cross_entropy_loss(self, Y_verif=None):
        if Y_verif is None:
            Y_hat = self.Y_train
            self.feed_forward(self.X_train)
            Y = self.layers[-1].axons_outputs
            # y = np.array([self.inference(x) for x in self.X_train])
            m = self.Y_train.shape[1]
        else:
            Y_hat = self.Y_verif
            self.feed_forward(self.X_verif)
            Y = self.layers[-1].axons_outputs
            # y = np.array([self.inference(x) for x in self.X_verif])
            m = self.Y_verif.shape[1]
        a = np.multiply( np.log(Y), Y_hat)
        b = np.multiply( np.log(1-Y), 1-Y_hat)
        sum_a = np.sum( a )
        sum_b = np.sum( b )
        result =  -(1./m) * ( sum_a + sum_b )
        return result

class Layer:
    def __init__(self, neurons_count, biases, weights_mat, previous_layer_neurons_count = 0, activation_function = 'sigmoid'):
        self.neurons_count = neurons_count
        self.biases = np.zeros((neurons_count, 1)) if biases is None else biases
        if weights_mat is None:
            weights_mat = np.random.randn(neurons_count, previous_layer_neurons_count) * np.sqrt(1. / neurons_count)
        self.weights = weights_mat
        self.neurons = [ Neuron(weights_mat[i], self.biases[i]) for i in range(neurons_count) ]
        self.activation_function = activation_function
        self.synaps_inputs_sums = np.zeros([neurons_count, 1])
        self.axons_outputs = np.zeros([neurons_count, 1])
        self.dW = np.zeros([neurons_count, previous_layer_neurons_count])
        self.db = np.zeros([neurons_count, 1])
        self.V_dW = np.zeros([neurons_count, previous_layer_neurons_count])
        self.V_db = np.zeros([neurons_count, 1])

    def print_myself(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights.shape)):
                print('  Weight:', self.neurons[n].weights[w])
                print('  Bias:  ', self.biases[w])

    def inference(self, inputs):
        self.inputs = inputs
        # y??? = ??(w???.T * x??? + b???)
        self.axons_outputs = np.array([neuron.axon_output(inputs) for neuron in self.neurons])
        return self.axons_outputs

    def compute_outputs(self, input_matrix):
        # y = ??(w.T * X + b)
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
        self.synaps_inputs_sums = np.matmul(self.weights, input_matrix) + self.biases
        self.axons_outputs = np.exp(self.synaps_inputs_sums) / np.sum(np.exp(self.synaps_inputs_sums), axis=0)
        return self.axons_outputs
    
    def sigmoid(self, sum_of_inputs):
        return 1 / (1 + np.exp(-sum_of_inputs))
    def relu(self, sum_of_inputs):
        return 0 if sum_of_inputs < 0 else sum_of_inputs
    def tangensH(self, sum_of_input):
        return np.tanh(sum_of_input)

class Neuron:
    def __init__(self, weights, bias = 0, activation_function = 'sigmoid'):
        self.weights = np.array(weights)
        self.activation_function = activation_function
        self.bias = bias

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
    
    # # ???E/???z??? -->  the partial derivative of the neuron error with respect to the neuron input
    # # ?? = ???E/???z??? = ???E/???y??? * dy???/dz???
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
    # # ???E/???y??? = 2 * 0.5 * (????? - y???) ^ (2 - 1) * -1 = 
    # #        = -(????? - y???) = -(desired output - actual output)
    # def DsynapsError_Doutput(self, desired_output):
    #     return -(desired_output - self.output)

    # # The derivative of the neuron output with respect to the total network input (sigmoid)
    # #      y??? = 1 / (1 + e^(-z???))  | d/dz???
    # # dy???/dz??? = y??? * (1 - y???)
    # def Doutput_DsynapsInputsSum_sigmoid(self):
    #     return self.output * (1 - self.output)

    # # The derivative of the neuron output with respect to the total network input (relu)
    # #      y??? = max(0, z???)  | d/dz???
    # # dy???/dz??? = [1 if z??? > 0 else 0]
    # def Doutput_DsynapsInputsSum_relu(self):
    #     return 1 if self.output > 0 else 0

    # # The derivative of the neuron output with respect to the total network input (tanh)
    # #      y??? = tanh(z???)  | d/dz???
    # # dy???/dz??? = 1 - (tanh(z???))^2
    # # dy???/dz??? = 1 - y???^2
    # def Doutput_DsynapsInputsSum_tanh(self):
    #     return 1 - self.output ** 2

    # # Each neuron error is calculated like this:
    # # E = ????? - y???
    # def error(self, desired_output):
    #     return 0.5 * (desired_output - self.output) ** 2

    # # The partial derivative of the total net input with respective to a given 
    # #     weight (with everything else held constant):
    # # ???z???/???w??? = some constant + 1 * x???w???^(1-0) + some constant ... = x???
    # def DsynapsInputsSum_Dweight(self, index):
    #     return self.inputs[index]



def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-s', '--seed', dest='seed', type=int,
                        default=123, help='type of algorithm')

    parser.add_argument('-t', '--type', dest='type', choices=('classification', 'regression'),
                        default='classification', help='type of algorithm')

    parser.add_argument('-f', '--file', dest='file', type=str, default="simple",
                        help='name of file')

    parser.add_argument('-n', '--number', dest='number', type=str, choices=('100', '500', '1000', "10000"),
                        default = "10000", help='count of samples')

    args = vars(parser.parse_args())

    print('Configuration')
    print('seed: {}'.format(args['seed']))
    print('algoritm type: {}'.format(args['type']))
    print('filename: {}'.format(args['file']))
    print('number of samples: {}'.format(args['number']))

    #Set seed
    np.random.seed(args['seed'])

    try:
        with open('network_params.json') as f:
            network_params = json.load(f)
    except:
        print("No network_paramss.json found")
        exit()

    type(network_params)

    # try:
    #     print("weight =", network_params["layers"][0]["weights"][0][1])
    # except:
    #     print("No such name in json file")
    # print(network_params.keys())

    print("Loading train and test files")
    data_raw = np.genfromtxt('out/mnist_784.csv', delimiter=',', skip_header = 1, usecols = range(0,785))
    shuffle_index = np.random.permutation(data_raw.shape[0])
    data = data_raw[shuffle_index]
    X = data[:,:784]/255
    y = data[:,784].astype(int)
    print("importing dataset...done!")

    net = Networkm(training_inputs=X, training_outputs=y)
    net.add_first_hidden(64, 784)
    net.add(10)
    # net.print_myself()
    # print("net.error({})={}".format([-0.432234621141106, 0.835330969654024], net.error([[1.]], [[0.8069052718966593, 0.7987629103901848]])))
    # print("net.total_error()={}".format(net.total_error()))
    # print("net.cross_entropy_loss()={}".format(net.cross_entropy_loss()))
    learning_rate = 4
    relaxation = .1
    batch_size = 128
    train_verif_ratio = .8
    net.train(batch_size, train_verif_ratio, learning_rate, relaxation)
    # net.print_myself()

    if args["type"] == "Regression":
        pass

    






    

if __name__ == "__main__":
    main()
