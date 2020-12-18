import pickle
import matplotlib.pyplot as plt
import numpy as np


def plot_cooling_schedules():
    lin_results = pickle.load(open('results/lin_results.pkl', 'rb'))
    log_results = pickle.load(open('results/log_results.pkl', 'rb'))
    lin_staged_results = pickle.load(open('results/lin_staged_results.pkl', 'rb'))
    log_mod_results = pickle.load(open('results/log_mod_results.pkl', 'rb'))

    lin_mean = np.mean(lin_results, axis=0)
    lin_std = np.std(lin_results, axis=0)

    log_mean = np.mean(log_results, axis=0)
    log_std = np.std(log_results, axis=0)

    lin_staged_mean = np.mean(lin_staged_results, axis=0)
    lin_staged_std = np.std(lin_staged_results, axis=0)

    log_mod_mean = np.mean(log_mod_results, axis=0)
    log_mod_std = np.std(log_mod_results, axis=0)

    plt.plot(lin_mean, label='lin')
    plt.fill_between(range(101), lin_mean + lin_std, lin_mean - lin_std, alpha=0.5)
    plt.plot(lin_staged_mean, label='lin (staged)')
    plt.fill_between(range(101), lin_staged_mean + lin_staged_std, lin_staged_mean - lin_staged_std, alpha=0.5)
    plt.plot(log_mean, label='log')
    plt.fill_between(range(101), log_mean + log_std, log_mean - log_std, alpha=0.5)
    plt.plot(log_mod_mean, label='log (modified)')
    plt.fill_between(range(101), log_mod_mean + log_mod_std, log_mod_mean - log_mod_std, alpha=0.5)

    plt.xlim(0, 100)
    plt.xlabel('# steps')
    plt.ylabel('Path length')
    plt.legend(title='cooling schedule')
    plt.tight_layout()
    plt.plot()
    plt.savefig('cooling_sched.png')
    plt.show()


def plot_starting_temp():
    starting_temp_results = pickle.load(open('results/starting_temp_results.pkl', 'rb'))

    init_temp_powers = [2, 1.8, 1.5, 1, 0]

    for i, temp_power in enumerate(init_temp_powers):
        mean = np.mean(starting_temp_results[i], axis=0)
        std = np.std(starting_temp_results[i], axis=0)

        plt.plot(mean, label=str(int(10 ** temp_power)))
        plt.fill_between(range(101), mean + std, mean - std, alpha=0.5)

    plt.xlim(0, 100)
    plt.xlabel('# steps')
    plt.ylabel('Path length')
    plt.legend(title=r'$T_0$')
    plt.tight_layout()
    plt.plot()
    plt.savefig('starting_temp.png')
    plt.show()


def plot_markov_chain_length():
    markov_results = pickle.load(open('results/markov_results.pkl', 'rb'))

    chain_lengths = np.logspace(3, 2, 4)

    for i, chain_length in enumerate(chain_lengths):
        mean = np.mean(markov_results[i], axis=0)
        print(mean[-1], chain_length)
        std = np.std(markov_results[i], axis=0)

        plt.plot(mean, label=str(int(chain_length)))
        plt.fill_between(range(101), mean + std, mean - std, alpha=0.5)

    plt.xlim(0, 100)
    plt.xlabel('# steps')
    plt.ylabel('Path length')
    plt.legend(title='MC length')
    plt.tight_layout()
    plt.plot()
    plt.savefig('MC_len.png')
    plt.show()


def plot_reorder_methods():
    switch_results = pickle.load(open('results/switch_results.pkl', 'rb'))
    move_results = pickle.load(open('results/move_results.pkl', 'rb'))

    mean_switch = np.mean(switch_results, axis=0)
    std_switch = np.std(switch_results, axis=0)
    mean_move = np.mean(move_results, axis=0)
    std_move = np.std(move_results, axis=0)

    plt.plot(mean_switch, label='switch')
    plt.fill_between(range(101), mean_switch + std_switch, mean_switch - std_switch, alpha=0.5)
    plt.plot(mean_move, label='move')
    plt.fill_between(range(101), mean_move + std_move, mean_move - std_move, alpha=0.5)

    plt.xlim(0, 100)
    plt.xlabel('# steps')
    plt.ylabel('Path length')
    plt.legend()
    plt.tight_layout()
    plt.plot()
    plt.savefig('reorder-methods.png')
    plt.show()


def plot_two_opt_first():
    log_results = pickle.load(open('results/log_results.pkl', 'rb'))
    two_opt_results = pickle.load(open('results/two_opt_results.pkl', 'rb'))

    mean_to = np.mean(two_opt_results, axis=0)
    std_to = np.std(two_opt_results, axis=0)
    log_mean = np.mean(log_results, axis=0)
    log_std = np.std(log_results, axis=0)
    plt.plot(mean_to, label='2-opt first')
    plt.fill_between(range(102), mean_to + std_to, mean_to - std_to, alpha=0.5)
    plt.plot(log_mean, label='default')
    plt.fill_between(range(101), log_mean + log_std, log_mean - log_std, alpha=0.5)

    plt.xlim(0, 100)
    plt.xlabel('# steps')
    plt.ylabel('Path length')
    plt.legend()
    plt.tight_layout()
    plt.plot()
    plt.show()


def plot_best():
    best_results = pickle.load(open('results/best_results.pkl', 'rb'))

    if len(best_results) > 1:
        mean_best = np.mean(best_results, axis=0)
        std_best = np.std(best_results, axis=0)
        plt.fill_between(range(101), mean_best + std_best, mean_best - std_best, alpha=0.5)
    else:
        mean_best = best_results[0]

    print(mean_best[0], mean_best[1], mean_best[-1])
    plt.plot(mean_best, label='two-opt')

    plt.xlim(0, 200)
    plt.xlabel('# steps')
    plt.ylabel('Path length')
    plt.legend()
    plt.tight_layout()
    plt.plot()
    plt.show()


if __name__ == "__main__":
    # plot_cooling_schedules()
    # plot_starting_temp()
    plot_markov_chain_length()
    # plot_reorder_methods()
    # plot_two_opt_first()
    # plot_best()
