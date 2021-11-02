import numpy as np
import itertools
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
regression_cost_funcions = ['pseudo_huber_loss', 'mean_squared_error']

classyfication_all = [
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
all_cases_classyfication = list(itertools.product(*classyfication_all))
all_cases_regression = list(itertools.product(*regression_all))


import subprocess
import time

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
# '-q', '--quit' fixed
import os
os.system('cd out; ls; cd -')
np.savetxt("out/continue_if_1.csv", np.array([1]), delimiter=",", header="CONTINUE_IF_1_BREAK_IF_0", comments='')
processes = []
for i, case in enumerate(all_cases_classyfication):
    if np.genfromtxt("out/continue_if_1.csv", delimiter=",", skip_header = 1) == 0:
        break
    cmd = 'mkdir -p experiments/data.' + case[2] + '.train.' + case[6] \
        + '; cd experiments/data.' + case[2] + '.train.' + case[6] \
        + '; python ../../main.py' \
        + ' -a {}'.format(case[0]) \
        + ' -c {}'.format(case[1]) \
        + ' -F {}'.format(case[2]) \
        + ' -H {}'.format(case[3]) \
        + ' -l {}'.format(case[4]) \
        + ' -n {}'.format(case[5]) \
        + ' -p {}'.format('classification') \
        + ' -P {}'.format(i) \
        + ' -r {}'.format(1) \
        + ' -s {}'.format(123) \
        + ' -S {}'.format(case[6]) \
        + ' -q {}'.format(1) \
        + '; cd -'
    print(cmd)
    p = subprocess.Popen(cmd, shell=True)
    processes.append(p)
    time.sleep(10)
    # out, err = p.communicate() 
    # result = out.split('\n')
    # for lin in result:
    #     if not lin.startswith('#'):
    #         print('lin =', lin)
for i, case in enumerate(all_cases_regression):
    if np.genfromtxt("out/continue_if_1.csv", delimiter=",", skip_header = 1) == 0:
        break
    cmd = 'mkdir -p experiments/data.' + case[2] + '.train.' + case[6] \
        + '; cd experiments/data.' + case[2] + '.train.' + case[6] \
        + '; python ../../main.py' \
        + ' -a {}'.format(case[0]) \
        + ' -c {}'.format(case[1]) \
        + ' -F {}'.format(case[2]) \
        + ' -h {}'.format(case[3]) \
        + ' -l {}'.format(case[4]) \
        + ' -n {}'.format(case[5]) \
        + ' -p {}'.format('regression') \
        + ' -P {}'.format(i) \
        + ' -r {}'.format(1) \
        + ' -s {}'.format(123) \
        + ' -S {}'.format(case[6]) \
        + ' -q {}'.format(1) \
        + '; cd -'
    print(cmd)
    p = subprocess.Popen(cmd, shell=True)
    processes.append(p)
    time.sleep(10)
print('len(all_cases_classyfication) =', len(all_cases_classyfication))
print('len(all_cases_regression) =', len(all_cases_regression))
print('processes started!')