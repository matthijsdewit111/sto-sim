# Names and UvAIDs:
# Matthijs de Wit, 10628258
# Menno Bruin, 11675225
#
# This file is used for measuring the variance reduction for the methods implemented in assignment1.py

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from assignment1 import (antithetic_sampling_method,
                         control_variate_sampling_method,
                         random_sampling_method)

if __name__ == "__main__":

    real_bounds = [-2, 0.5]
    imag_bounds = [-1.25, 1.25]

    cv_circles = [
        {
            "c": [-0.15, 0],
            "r": 0.60
        },
        {
            "c": [-1.0, 0],
            "r": 0.25
        }
    ]
    cv = -0.9248455094960547

    final_results_normal_std = []
    final_results_normal_mean = []
    final_results_normal_var = []
    final_results_antithetic_std = []
    final_results_antithetic_mean = []
    final_results_antithetic_var = []
    samples = list(range(25, 40026, 2000))
    for s in tqdm(samples):
        results_normal = []
        results_antithetic = []
        for _ in range(25):
            a_normal = random_sampling_method(int(s), real_bounds, imag_bounds, int(1e3), False)
            a_antithetic = control_variate_sampling_method(int(s), real_bounds, imag_bounds, int(1e3), cv_circles, cv, False)
            # a_antithetic = antithetic_sampling_method(int(s), real_bounds, imag_bounds, int(1e3), False)

            results_normal.append(a_normal)
            results_antithetic.append(a_antithetic)

        final_results_normal_mean.append(np.mean(results_normal))
        final_results_normal_std.append(np.std(results_normal))
        final_results_normal_var.append(np.var(results_normal))
        final_results_antithetic_mean.append(np.mean(results_antithetic))
        final_results_antithetic_std.append(np.std(results_antithetic))
        final_results_antithetic_var.append(np.var(results_antithetic))

    final_results_normal_mean = np.array(final_results_normal_mean)
    final_results_normal_std = np.array(final_results_normal_std)

    final_results_antithetic_mean = np.array(final_results_antithetic_mean)
    final_results_antithetic_std = np.array(final_results_antithetic_std)

    plt.plot(samples, final_results_normal_mean, label="Pure Random")
    plt.fill_between(samples, final_results_normal_mean - final_results_normal_std,
                     final_results_normal_mean + final_results_normal_std,
                     alpha=0.5)

    plt.plot(samples, final_results_antithetic_mean, label="Control Variates", color='palegreen')
    plt.fill_between(samples, final_results_antithetic_mean - final_results_antithetic_std,
                     final_results_antithetic_mean + final_results_antithetic_std,
                     alpha=0.5, color='palegreen')

    plt.xlabel("# Samples")
    plt.ylabel("Surface Area")
    plt.xlim(0, 40000)
    plt.ylim(1.4, 1.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # plt.plot(np.array(final_results_normal_var) - np.array(final_results_antithetic_var))

    # plt.show()
