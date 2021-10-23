import json
import argparse
import random as rand
from numpy import random
from math import exp, tanh

from utils import read_csv_file

class Network:
    def __init__(self):
        self.layers_count = 1
        self.layers = []

    def add_first_hidden(self, neurons_count, inputs_count, bias = 0, weights_mat = None):
        self.layers.append(Layer(neurons_count, bias, weights_mat, inputs_count))
        self.layers_count += 1

    def add(self, neurons_count, bias = 0, weights_mat = None):
        self.layers.append(Layer(neurons_count, bias, weights_mat, self.layers[self.layers_count - 1].neurons_count))
        self.layers_count += 1

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
        return self.output_layer.inference(self.outputs)
    
    def train():
        pass

    def total_error(self, training_sets):
        error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.inference(training_inputs)
            for o in range(len(training_outputs)):
                error += self.layers[-1].neurons[o].calculate_error(training_outputs[o])
        return error




class Layer:
    def __init__(self, neurons_count, bias, weights_mat, previous_layer_neurons_count = 0):
        self.neurons_count = neurons_count
        self.bias = bias if bias else rand.random()
        if weights_mat is None:
            weights_mat = random.rand(neurons_count, previous_layer_neurons_count)
        self.neurons = [ Neuron(neuron_weights, self.bias) for neuron_weights in weights_mat ]

    def print_myself(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:  ', self.bias)

    def inference(self, inputs):
        return [neuron.axon_output(inputs) for neuron in self.neurons]


class Neuron:
    def __init__(self, weights, activation_function = 'sigmoid', bias = 0):
        self.weights = weights
        self.activation_function = activation_function
        self.bias = bias

    def axon_output(self, inputs):
        self.inputs = inputs
        if self.activation_function == 'sigmoid':
            self.output = self.sigmoid(self.synaps_inputs_sum())
        elif self.activation_function == 'relu':
            self.output = self.relu(self.synaps_inputs_sum())
        elif self.activation_function == 'tanh':
            self.output = self.tangensH(self.synaps_inputs_sum())
        else:
            print("No such activation_function:", self.activation_function)
            exit()
    
    def sigmoid(sum_of_inputs):
        return 1 / (1 + exp(-sum_of_inputs))
    def relu(sum_of_inputs):
        return 0 if sum_of_inputs < 0 else sum_of_inputs
    def tangensH(sum_of_input):
        return tanh(sum_of_input)

    def synaps_inputs_sum(self):
        sum = 0
        for i in range(len(self.inputs)):
            sum += self.inputs[i] * self.weights[i]
        return sum + self.bias
    
    # ∂E/∂zⱼ -->  the partial derivative of the neuron error with respect to the neuron input
    # δ = ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ
    def DsynapsError_DsynapsInputsSum():
        pass

    # The partial derivative of the neuron error with respect to actual neuron output:
    # ∂E/∂yⱼ = 2 * 0.5 * (target output - actual output) ^ (2 - 1) * -1 = -(tⱼ - yⱼ) = -(target output - actual output)
    def DsynapsError_Doutput(self, target_output):
        return -(target_output - self.output)

    # The derivative of the neuron output with respect to the total network input
    #      yⱼ = 1 / (1 + e^(-zⱼ))  | d/dzⱼ
    # dyⱼ/dzⱼ = yⱼ * (1 - yⱼ)
    def Doutput_DsynapsInputsSum(self):
        return self.output * (1 - self.output)

    # E = yⱼ* - yⱼ
    def network_error(self, desired_output):
        return 0.5 * (desired_output - self.output)



def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-s', '--seed', dest='seed', type=int,
                        default=123, help='type of algorithm')

    parser.add_argument('-t', '--type', dest='type', choices=('classification', 'regression'),
                        default='classification', help='type of algorithm')

    parser.add_argument('-f', '--file', dest='file', type=str, default="simple",
                        help='name of file')

    parser.add_argument('-n', '--number', dest='number', type=str, choices=('100', '500', '1000', "10000"),
                        default = "100", help='count of samples')

    args = vars(parser.parse_args())

    print('Configuration')
    print('seed: {}'.format(args['seed']))
    print('algoritm type: {}'.format(args['type']))
    print('filename: {}'.format(args['file']))
    print('number of samples: {}'.format(args['number']))

    #Set seed
    random.seed(args['seed'])

    try:
        with open('network_params.json') as f:
            network_params = json.load(f)
    except:
        print("No network_paramss.json found")
        exit()

    type(network_params)

    network_params.keys()
    try:
        print(network_params["layers"][0]["weights"][0][1])
    except:
        print("No such name in json file")
        
    print(network_params.keys())

    print("Loading train and test files")

    #Load train file
    train_file = read_csv_file(args["type"], args["file"], 'train', args["number"])

    #Load test file
    test_file = read_csv_file(args["type"], args["file"], 'test', args["number"])

    X_train = train_file.x
    y_train = train_file.y

    X_test = test_file.x
    y_test = test_file.y

    if args["type"] == "classification":
        train_cls = train_file.cls
        test_cls = test_file.cls

    






    

if __name__ == "__main__":
    main()