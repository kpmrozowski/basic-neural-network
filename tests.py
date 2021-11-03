import numpy as np
import itertools
from experiment import experiment
import subprocess
import time
import logging
import threading




from multiprocessing.pool import ThreadPool
from random import randrange
from time import sleep
import numpy as np
import pandas as pd

def foo(args):
    return [args['process']/7., args['process']**2/7.]

if __name__ == '__main__':

    # tests:
    file_sizes = np.array([100, 500, 1000, 10000])
    learning_rates = np.array([1e-4, .1, .5]) # np.array([ 2 ** i for i in range(-4,6) ])
    # relaxations = np.array([ 0.05, 0.1, 0.2, 0.4, 0.8 ]), default=0.1
    # huber_delta = np.array([ 0.05, 0.1, 0.2, 0.4, 0.8 ]), default=1.0
    numbers_of_hidden_layers = np.array([ 0, 1, 2, 3, 4 ])
    numbers_of_neurons_in_layer = np.array([ 4, 16, 64 ])
    activation_functions = [ 'sigmoid', 'relu', 'tanh' ]
    classification_file_types = [ 'simple', 'three_gauss' ]
    regression_file_types = [ 'activation', 'cube' ]
    classification_cost_funcions = ['cross_entropy', 'mean_squared_error']
    regression_cost_funcions = ['pseudo_huber', 'mean_squared_error']

    classification_all = [
        activation_functions,
        classification_cost_funcions,
        classification_file_types,
        numbers_of_hidden_layers,
        learning_rates,
        numbers_of_neurons_in_layer,
        file_sizes,

    ]
    regression_all = [
        activation_functions,
        regression_cost_funcions,
        regression_file_types,
        numbers_of_hidden_layers,
        learning_rates,
        numbers_of_neurons_in_layer,
        file_sizes,

    ]

    # generate_experiments:
    all_cases_classification = list(itertools.product(*classification_all))
    all_cases_regression = list(itertools.product(*regression_all))

    print('len(all_cases_classification) =', len(all_cases_classification))
    print('len(all_cases_regression) =', len(all_cases_regression))


    # Arguments:
    # '-a', '--activation' [0]
    # '-c', '--cost' [1]
    # '-F', '--file' [2]
    # '-h', '--hidden-layers' [3]
    # '-l', '--learning-rate' [4]
    # '-n', '--neurons' [5]
    # '-p', '--problem' const
    # '-P', '--process' fixed
    # '-s', '--seed' const
    # '-S', '--size' [6]
    # '-q', '--quit' const
    

    # threaded version
    pool_classification = ThreadPool(10)
    results_classification = []
    data_list_classification = []
    for i, case in enumerate(all_cases_classification):
        args = {
            "activation": case[0],
            "cost": case[1],
            "file": case[2],
            "hidden_layers": case[3],
            "learning_rate": case[4],
            "neurons": case[5],
            "problem": 'classification',
            "process": i,
            "remote": 1,
            "seed": 123,
            "size": case[6],
            "quit": 0
        }
        params = {
            "activation": case[0],
            "cost": case[1],
            "file": case[2],
            "hidden_layers": case[3],
            "learning_rate": case[4],
            "neurons": case[5],
            "problem": 'classification',
            "size": case[6],
            "accuracy_train": 0,
            "accuracy_test": 0
        }
        results_classification.append( pool_classification.apply_async(experiment, args=(args,)) )
        data_list_classification.append(params)
    pool_classification.close()
    pool_classification.join()
    

    results_classification = [ r.get() for r in results_classification ]
    for i, data_row in enumerate(data_list_classification):
        data_row["accuracy_train"] = results_classification[i][0]
        data_row["accuracy_test"] = results_classification[i][1]
        pass
    df_classification = pd.DataFrame(data_list_classification)
    print(df_classification.head())
    df_classification.to_csv('results_classification.csv', float_format='%.2E')

    pool_regression = ThreadPool(10)
    results_regression = []
    data_list_regression = []
    for i, case in enumerate(all_cases_regression):
        args = {
            "activation": case[0],
            "cost": case[1],
            "file": case[2],
            "hidden_layers": case[3],
            "learning_rate": case[4],
            "neurons": case[5],
            "problem": 'regression',
            "process": i + len(all_cases_classification),
            "remote": 1,
            "seed": 123,
            "size": case[6],
            "quit": 0
        }
        params = {
            "activation": case[0],
            "cost": case[1],
            "file": case[2],
            "hidden_layers": case[3],
            "learning_rate": case[4],
            "neurons": case[5],
            "problem": 'regression',
            "size": case[6],
            "accuracy_train": 0,
            "accuracy_test": 0
        }
        results_regression.append( pool_regression.apply_async(experiment, args=(args,)) )
        data_list_regression.append(params)

    pool_regression.close()
    pool_regression.join()
    results_regression = [ r.get() for r in results_regression ]
    # print('results: ', results)
    for i, data_row in enumerate(data_list_regression):
        data_row["accuracy_train"] = results_regression[i][0]
        data_row["accuracy_test"] = results_regression[i][1]
        pass
    df_regression = pd.DataFrame(data_list_regression)
    print(df_regression.head())
    df_regression.to_csv('results_regression.csv', float_format='%.2E')
    








# Subprocesses version
# np.savetxt("out/continue_if_1.csv", np.array([1]), delimiter=",", header="CONTINUE_IF_1_BREAK_IF_0", comments='')
# processes = []
# for i, case in enumerate(all_cases_classyfication):
#     if np.genfromtxt("out/continue_if_1.csv", delimiter=",", skip_header = 1) != 1:
#         break
#     cmd = 'mkdir -p experiments/classification/data.' + case[2] + '.train.' + str(case[6]) \
#         + '; cd experiments/classification/data.' + case[2] + '.train.' + str(case[6]) \
#         + '; python ../../../main.py' \
#         + ' -a {}'.format(case[0]) \
#         + ' -c {}'.format(case[1]) \
#         + ' -F {}'.format(case[2]) \
#         + ' -H {}'.format(case[3]) \
#         + ' -l {}'.format(case[4]) \
#         + ' -n {}'.format(case[5]) \
#         + ' -p {}'.format('classification') \
#         + ' -P {}'.format(i) \
#         + ' -r {}'.format(1) \
#         + ' -s {}'.format(123) \
#         + ' -S {}'.format(case[6]) \
#         + ' -q {}'.format(0) \
#         + '; cd -'
#     print(cmd)
#     threads = list()
#     for index in range(1):
#         logging.info("Main    : create and start thread %d.", index)
#         x = threading.Thread(target=experiment, args=(args,))
#         threads.append(x)
#         x.start()
#     p = subprocess.Popen(cmd, shell=True)
#     processes.append(p)


# for i, case in enumerate(all_cases_regression):
#     if np.genfromtxt("out/continue_if_1.csv", delimiter=",", skip_header = 1) != 1:
#         break
#     cmd = 'mkdir -p experiments/regression/data.' + case[2] + '.train.' + str(case[6]) \
#         + '; cd experiments/regression/data.' + case[2] + '.train.' + str(case[6]) \
#         + '; python ../../../main.py' \
#         + ' -a {}'.format(case[0]) \
#         + ' -c {}'.format(case[1]) \
#         + ' -F {}'.format(case[2]) \
#         + ' -H {}'.format(case[3]) \
#         + ' -l {}'.format(case[4]) \
#         + ' -n {}'.format(case[5]) \
#         + ' -p {}'.format('regression') \
#         + ' -P {}'.format(i) \
#         + ' -r {}'.format(1) \
#         + ' -s {}'.format(123) \
#         + ' -S {}'.format(case[6]) \
#         + ' -q {}'.format(0) \
#         + '; cd -'
#     print(cmd)
#     p = subprocess.Popen(cmd, shell=True)
#     processes.append(p)
#     time.sleep(10)
print('processes finished!')