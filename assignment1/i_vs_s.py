# Names and UvAIDs:
# Matthijs de Wit, 10628258
# Menno Bruin, 11675225
#
# This file is used for measuring the effects of the number of samples and the number of max_iterations

import matplotlib.pyplot as plt
import numpy as np

from assignment1 import random_sampling_method

if __name__ == "__main__":
    fig, axs = plt.subplots(2)

    real_A_M = 1.50659177

    s_list = np.logspace(2, 6, 20)
    i_list = np.logspace(2, 5, 20)

    X, Y = np.meshgrid(s_list, i_list)

    real_bounds = [-2, 0.5]
    imag_bounds = [-1.25, 1.25]

    s_std_results = []
    s_mean_results = []
    for s in s_list:
        individual_results = []
        for _ in range(10):
            a = random_sampling_method(int(s), real_bounds, imag_bounds, int(1e4), False)
            individual_results.append(a)
        s_std_results.append(np.std(individual_results))
        s_mean_results.append(np.mean(individual_results) - real_A_M)

    s_std_results = np.array(s_std_results)
    s_mean_results = np.array(s_mean_results)

    i_std_results = []
    i_mean_results = []
    for i in i_list:
        print(i)
        individual_results = []
        for _ in range(10):
            a = random_sampling_method(int(1e6), real_bounds, imag_bounds, int(i), False)
            individual_results.append(a)
        i_std_results.append(np.std(individual_results))
        i_mean_results.append(np.mean(individual_results) - real_A_M)

    i_std_results = np.array(i_std_results)
    i_mean_results = np.array(i_mean_results)

    axs[0].plot(s_list, s_mean_results)
    axs[0].fill_between(s_list, s_mean_results - s_std_results, s_mean_results + s_std_results, alpha=0.5)
    axs[0].set_xlabel('num_samples')
    axs[0].set_ylabel('deviation from real A_M')
    axs[0].set_xscale('log')
    axs[0].plot([0, max(s_list)], [0, 0], '--')  # show 0 line

    axs[1].plot(i_list, i_mean_results)
    axs[1].fill_between(i_list, i_mean_results - i_std_results, i_mean_results + i_std_results, alpha=0.5)
    axs[1].set_xlabel('max_iterations')
    axs[1].set_ylabel('deviation from real A_M')
    axs[1].set_xscale('log')
    axs[1].plot([0, max(i_list)], [0, 0], '--')  # show 0 line

    plt.tight_layout()
    plt.show()
