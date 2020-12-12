import pickle

import matplotlib.pyplot as plt
from tqdm import tqdm

from travelling_salesman.travelling_salesman import *
import os

import cProfile

base_path = 'travelling_salesman/TSP-Configurations/'
data_files = ['a280.tsp.txt', 'pcb442.tsp.txt', 'eil51.tsp.txt']


def read_tsp_file(file_name: str = data_files[0]) -> [Node]:
    file_path = base_path + file_name
    with open(file_path, 'r') as f:
        name = f.readline().strip().split()[1]  # NAME
        comment = f.readline().strip().split()[1]  # COMMENT
        file_type = f.readline().strip().split()[1]  # TYPE
        dimension = f.readline().strip().split()[1]  # DIMENSION
        edge_weight_type = f.readline().strip().split()[1]  # EDGE_WEIGHT_TYPE
        f.readline()
        nodes = [from_listofstrings(node) for node in [line.strip().split() for line in f.readlines()[:-1]]]
        f.close()
    return nodes


def simulated_annealing(Ts, markov_chain_length, method='switch', two_opt_first=False, file_name=data_files[0]):
    nodes = read_tsp_file(file_name)
    ts = TravellingSalesman(nodes)

    total_distances = [ts.get_total_distance()]
    # print("starting distance:", total_distances[0])

    if two_opt_first:
        # print('applying 2-opt to intersections...')
        ts.two_opt()
        total_distances.append(ts.get_total_distance())
        # print("distance after 2-opt:", total_distances[1])

    for i, T in enumerate(Ts):
        for _ in range(int(markov_chain_length)):
            before_length = ts.get_total_distance()

            if method == 'switch':
                a, b, c, d = ts.switch()
            elif method == 'move':
                a, b, c = ts.move()
            else:
                raise Exception('invalid method')

            after_length = ts.get_total_distance()

            if after_length > before_length:  # if new solution is worse
                p = np.exp(-(after_length - before_length) / T)
                r = random.random()
                if r > p:
                    if method == 'switch':
                        ts._switch_edges(a, b, c, d, revert=True)
                    elif method == 'move':
                        ts._move_node(a, b, c, revert=True)

        total_distances.append(ts.get_total_distance())

    # print("final distance:", total_distances[-1])
    return ts, total_distances


def test_scheduling_strategies():
    Ts_lin = np.linspace(50, 1, 100)
    Ts_log = np.logspace(1.5, -2, 100)
    Ts_lin_staged = np.hstack((np.linspace(50, 2, 50), np.linspace(1, 0.1, 50)))  # 100 to 2 and 1 to 0.1
    TS_log_modified = [20 / np.log(n + 1.5) - 4.3 for n in range(0, 100)]

    markov_chain_length = 1000

    lin_results = []
    log_results = []
    lin_staged_results = []
    log_mod_results = []

    repetitions = 100
    for _ in tqdm(range(repetitions)):
        _, total_distances_lin = simulated_annealing(Ts_lin, markov_chain_length)
        lin_results.append(total_distances_lin)

        _, total_distances_log = simulated_annealing(Ts_log, markov_chain_length)
        log_results.append(total_distances_log)

        _, total_distances_lin_staged = simulated_annealing(Ts_lin_staged, markov_chain_length)
        lin_staged_results.append(total_distances_lin_staged)

        _, total_distances_log_mod = simulated_annealing(TS_log_modified, markov_chain_length)
        log_mod_results.append(total_distances_log_mod)

    pickle.dump(lin_results, open('results/lin_results.pkl', 'wb'))
    pickle.dump(log_results, open('results/log_results.pkl', 'wb'))
    pickle.dump(lin_staged_results, open('results/lin_staged_results.pkl', 'wb'))
    pickle.dump(log_mod_results, open('results/log_mod_results.pkl', 'wb'))


def test_starting_temp():
    markov_chain_length = 1000

    init_temp_powers = [2, 1.8, 1.5, 1, 0]

    starting_temp_results = [[], [], [], [], []]

    repetitions = 100
    for _ in tqdm(range(repetitions)):
        for i, T_init_power in enumerate(init_temp_powers):
            Ts = np.logspace(T_init_power, -2, 100)
            _, total_distances = simulated_annealing(Ts, markov_chain_length)
            starting_temp_results[i].append(total_distances)

    pickle.dump(starting_temp_results, open('results/starting_temp_results.pkl', 'wb'))


def test_markov_chain_length():
    Ts = np.logspace(1.5, -2, 100)

    chain_lenghts = np.logspace(4, 2, 7)

    markov_results = [[], [], [], [], [], [], []]

    repetitions = 100
    for _ in tqdm(range(repetitions)):
        for i, markov_chain_length in enumerate(chain_lenghts):
            _, total_distances = simulated_annealing(Ts, markov_chain_length)
            markov_results[i].append(total_distances)

    pickle.dump(markov_results, open('results/markov_results.pkl', 'wb'))


def test_reorder_methods():
    Ts = np.logspace(1.5, -2, 100)
    markov_chain_length = 1000

    switch_results = []
    move_results = []

    repetitions = 100
    for _ in tqdm(range(repetitions)):
        _, total_distances_switch = simulated_annealing(Ts, markov_chain_length, method='switch')
        switch_results.append(total_distances_switch)
        _, total_distances_move = simulated_annealing(Ts, markov_chain_length, method='move')
        move_results.append(total_distances_move)

    pickle.dump(switch_results, open('results/switch_results.pkl', 'wb'))
    pickle.dump(move_results, open('results/move_results.pkl', 'wb'))


def test_two_opt_first():
    Ts = np.logspace(1, -2, 100)
    markov_chain_length = 1000

    two_opt_results = []

    repetitions = 100
    for _ in tqdm(range(repetitions)):
        _, total_distances = simulated_annealing(Ts, markov_chain_length, method='switch')
        two_opt_results.append(total_distances)

    pickle.dump(two_opt_results, open('results/two_opt_results.pkl', 'wb'))


if __name__ == '__main__':
    Ts = np.logspace(1.5, -2, 100)  # use with two_opt

    # print(Ts)
    markov_chain_length = 10

    # ts, total_distances = simulated_annealing(Ts, markov_chain_length, method='move', two_opt_first=True, file_name=data_files[1])

    # ts.draw_graph(draw_edges=True)

    # plt.plot(total_distances)
    # plt.show()

    if not os.path.exists('results'):
        os.mkdir('results')

    # cProfile.run('simulated_annealing(Ts, markov_chain_length)', sort='cumtime')

    # test_scheduling_strategies()
    # test_starting_temp()
    # test_markov_chain_length()
    # test_reorder_methods()
    # test_two_opt_first()
