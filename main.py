from Network import Network
from mnist import Networkm
import json
import argparse
from posix import XATTR_SIZE_MAX
import numpy as np
from matplotlib import pyplot, colors
from utils import read_csv_file, comp_confmat, plot_set, plot_classification, read_csv_file_test #, DrawNN
import time


def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-a', '--activation', dest='activation', type=str, 
                        choices=('sigmoid', 'relu', 'tanh'),
                        default='sigmoid', help='type of cost function')

    parser.add_argument('-c', '--cost', dest='cost', type=str, 
                        choices=('cross_entropy', 'mean_squared_error', 'pseudo_huber'),
                        default='mean_squared_error', help='type of cost function')

    parser.add_argument('-F', '--file', dest='file', type=str, 
                        choices=('simple', 'three_gauss', 'circles', 'XOR', 'noisyXOR',
                         'activation', 'cube', 'linear', 'multimodal', 'square', 'mnist'), 
                        default="circles", help='name of file')

    parser.add_argument('-H', '--hidden-layers', dest='hidden_layers', type=int,
                        default=1, help='number of hidden layers')

    parser.add_argument('-l', '--learning-rate', dest='learning_rate', type=float,
                        default=.1, help='type of algorithm')

    parser.add_argument('-n', '--neurons', dest='neurons', type=int,
                        default=5, help='number of neuron in every layer')

    parser.add_argument('-p', '--problem', dest='problem', type=str, 
                        choices=('classification', 'regression'),
                        default='classification', help='type of algorithm')

    parser.add_argument('-P', '--process', dest='process', type=int,
                        default=0, help='process number')

    parser.add_argument('-r', '--remote', dest='remote', type=int,
                        default=0, help='remote control')

    parser.add_argument('-s', '--seed', dest='seed', type=int,
                        default=123, help='type of algorithm')

    parser.add_argument('-S', '--size', dest='size', type=str,
                        choices=('100', '500', '1000', "10000"),
                        default = "10000", help='count of samples')

    parser.add_argument('-q', '--quit', dest='quit', type=int,
                        default=0, help='quit immediately')
    
    args = vars(parser.parse_args())
    
    print('{}_started'.format(args['process']))

    if args["quit"] == 1:
        print("This proces: {}, remote={}".format(args["process"], args["remote"]))
        time.sleep(1)
        print('process_{}'.format(args['process']))
        exit()

    if False:
        print('Configuration')
        print('seed: {}'.format(args['seed']))
        print('machine learning problem type: {}'.format(args['problem']))
        print('filename: {}'.format(args['file']))
        print('number of samples: {}'.format(args['size']))

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

    if args['remote'] == 0:
        print("Loading train and test files")
    # train_file = read_csv_file(args["problem"], args["file"], 'train', args["size"])
    # test_file = read_csv_file(args["problem"], args["file"], 'test', args["size"])
    train_file = read_csv_file_test(args["problem"], args["file"], 'train', args["size"])
    test_file = read_csv_file_test(args["problem"], args["file"], 'test', args["size"])
    shuffle_index = np.random.permutation(train_file.shape[0])
    train_file = train_file[shuffle_index]

    # print("Dim: {}\tHead:".format(train_file.shape))
    # print(train_file.head())

    # train_file_X = train_file.x
    # train_file_y = train_file.y
    # test_file_X = test_file.x
    # test_file_y = test_file.y

    if args["file"] == 'mnist':
        shuffle_index = np.random.permutation(train_file.shape[0])
        data = train_file[shuffle_index]
        train_set_X = data[:,:784]/255
        train_set_Y = data[:,784].astype(int)
        print("importing dataset...done!")

        net = Networkm(training_inputs=train_set_X, training_outputs=train_set_Y, activation_function='sigmoid', problem='classification')
        net.add_first_hidden(64, 784)
        net.add(10)
        learning_rate = 4
        relaxation = .1
        batch_size = 128
        train_verif_ratio = .8
        net.train(batch_size, train_verif_ratio, learning_rate, relaxation)
    
    elif args["problem"] == "classification":
        shuffle_index_train = np.random.permutation(train_file.shape[0])
        shuffle_index_test = np.random.permutation(test_file.shape[0])
        train_file = train_file[shuffle_index_train]
        test_file  = test_file[shuffle_index_test]
        if True:
            plot_set(train_file)
            plot_set(test_file)
        train_set_coords = train_file[:,:-1]
        train_cls = train_file[:,-1].astype(int)
        test_set_coords = test_file[:,:-1]
        test_cls = test_file[:,-1].astype(int)

        min_cls = np.min(train_cls)
        max_cls = np.max(train_cls)
        classes_count = max_cls - min_cls + 1


        if args["remote"] == 1:     
            cost_function = args["cost"]
            activation_function = args["activation"]
            problem = args["problem"]
            learning_rate = args["learning_rate"]
            huber_delta = 2.
            net = Network(
                training_inputs=train_set_coords, 
                training_outputs=train_cls, 
                cost_function=cost_function, 
                activation_function=activation_function, 
                problem=problem )
            if args["hidden_layers"] == 0:
                net.add_first_hidden(classes_count, train_set_coords.shape[1])
            else:
                net.add_first_hidden(args["neurons"], train_set_coords.shape[1])
                for _ in range(args["hidden_layers"] - 1):
                    net.add(args["neurons"])
                net.add(classes_count)
        else:     
            cost_function = 'cross_entropy'
            activation_function = 'relu'
            problem = 'classification'
            learning_rate = 1e-4      # .1 for mean_squared_error      1e-4 for cross_entropy
            net = Network(
                training_inputs=train_set_coords, 
                training_outputs=train_cls, 
                cost_function=cost_function, 
                activation_function=activation_function, 
                problem=problem )
            net.add_first_hidden(classes_count**3, train_set_coords.shape[1])
            net.add(classes_count*3)
            net.add(classes_count*2)
            net.add(classes_count)
        relaxation = .1
        batch_size = 128
        train_verif_ratio = .8
        iterations_initial = 100
        timeout = time.time() + 60
        
        net.train(batch_size, train_verif_ratio, learning_rate, relaxation, timeout, iterations_initial, args['remote'])
        # net.print_myself()
        test_predictions = net.predict(test_set_coords)
        if args["remote"] == 0:
            print(comp_confmat(test_predictions, test_cls - 1))
            print('accuracy(test) =', np.mean(test_predictions == test_cls - 1))
        
        plot_classification(test_file, test_predictions)

    elif args["problem"] == "regression":
        train_set_coords = train_file[:,:-1][:,0]
        train_set_value = train_file[:,-1].astype(float)
        test_set_coords = test_file[:,:-1][:,0]
        test_set_value = test_file[:,-1].astype(float)

        # net.print_myself()

        # ustawienia:
        # iter  |   learning_rate   |   huber_delta
        #  1    |          0.5      |      2.00      
        #  2    |         16.0      |      0.09

        if args["remote"] == 1:     
            cost_function = args["cost"]
            activation_function = args["activation"]
            problem = args["problem"]
            learning_rate = args["learning_rate"]
            huber_delta = 2.
            net = Network(
                training_inputs=train_set_coords, 
                training_outputs=train_set_value, 
                cost_function=cost_function, 
                activation_function=activation_function, 
                problem=problem )
            if args["hidden_layers"] == 0:
                net.add_first_hidden(1, 1)
            else:
                net.add_first_hidden(args["neurons"], 1)
                for _ in range(args["hidden_layers"] - 1):
                    net.add(args["neurons"])
                net.add(1)
        else:     
            cost_function = 'pseudo_huber'
            activation_function = 'tanh'
            problem = 'regression'
            learning_rate = .5
            net = Network(
                training_inputs=train_set_coords, 
                training_outputs=train_set_value, 
                cost_function=cost_function, 
                activation_function=activation_function, 
                problem=problem )
            net.add_first_hidden(20, 1)
            net.add(20)
            net.add(1)
        relaxation = .1
        batch_size = 128
        train_verif_ratio = .8
        iterations_initial = 100
        huber_delta = 2.
        timeout = time.time() + 60

        # all results that we need:
        # results = {
        #     "train_cost": [],
        #     "verif_cost": [],
        #     "accuracy_train": [],
        #     "accuracy_verif": [] }
        #  + train_predictions + verif_predictions + test_predictions + decision_surface
        
        net.train(batch_size, train_verif_ratio, learning_rate, relaxation, timeout, iterations_initial, args['remote'], huber_delta)
        # net.print_myself()
        if args["remote"] == 0:
            X = np.divide( train_set_coords - net.bounds_raw[0], net.bounds_raw[1] - net.bounds_raw[0] )
            Y = np.divide( train_set_value - net.bounds_raw[2], net.bounds_raw[3] - net.bounds_raw[2] )
            X_train = np.array([X])
            Y_train = np.array([Y])
            X = np.divide( test_set_coords - net.bounds_raw[0], net.bounds_raw[1] - net.bounds_raw[0] )
            Y = np.divide( test_set_value - net.bounds_raw[2], net.bounds_raw[3] - net.bounds_raw[2] )
            X_test = np.array([X])
            Y_test = np.array([Y])
            if args["cost"] == 'cross_entropy':
                train_loss = net.cross_entropy_loss(X_train, Y_train)
                test_loss = net.cross_entropy_loss(X_test, Y_test)
            elif args["cost"] == 'pseudo_huber':
                train_loss = net.pseudo_huber_loss(X_train, Y_train)
                test_loss = net.pseudo_huber_loss(X_test, Y_test)
            elif args["cost"] == 'mean_squared_error':
                train_loss = net.mean_squared_error(X_train, Y_train)
                test_loss = net.mean_squared_error(X_test, Y_test)
            train_predictions = net.predict(train_set_coords)
            test_predictions = net.predict(test_set_coords)
            fig, (ax0, ax1) = pyplot.subplots(1,2)
            fig.suptitle('Final predictions!')
            l00 = ax0.scatter(train_set_coords, train_predictions, marker="o",  s=.5)
            l01 = ax0.scatter(train_set_coords, train_set_value, marker="o",  s=.5)
            l10 = ax1.scatter(test_set_coords, test_predictions, marker="o",  s=.5)
            l11 = ax1.scatter(test_set_coords, test_set_value, marker="o",  s=.5)
            ax0.set_title('Testining set, LOSS={:.4f}‰'.format(1000*train_loss))
            ax1.set_title('Testining set, LOSS={:.4f}‰'.format(1000*test_loss))
            ax0.set(xlabel='x', ylabel='y')
            ax1.set(xlabel='x', ylabel='y')
            pyplot.legend((l00, l01), ('Y_train', 'predictions'), loc='upper right', shadow=True)
            pyplot.legend((l10, l11), ('Y_test', 'predictions'), loc='upper right', shadow=True)
            pyplot.savefig('final-predictions.png', dpi=300)
            # pyplot.show()
    print('{}_finished'.format(args['process']))
    pass

    






    

if __name__ == "__main__":
    main()