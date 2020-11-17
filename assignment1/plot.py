import pickle as pk
import matplotlib.pyplot as plt
import numpy as np


with open('random_sampling_data', 'rb') as f:
    dict = pk.load(f)
    samples = dict['samples']
    means = dict['means']
    stds = dict['stds']

    plt.plot(samples, means, '-', label='Pure Random')
    plt.fill_between(samples, np.subtract(means, stds), np.add(means, stds), alpha=.2)

# with open('orthogonal_data', 'rb') as f:
#     dict = pk.load(f)
#     samples = dict['samples']
#     means = dict['means']
#     stds = dict['stds']
#
#     plt.plot(samples, means, '-', color='orange', label='Orthogonal LHS')
#     plt.fill_between(samples, np.subtract(means, stds), np.add(means, stds), alpha=.2, color='orange')

with open('lhs_data', 'rb') as f:
    dict = pk.load(f)
    samples = dict['samples']
    means = dict['means']
    stds = dict['stds']

    plt.plot(samples, means, '-', label='LHS', color='green')
    plt.fill_between(samples, np.subtract(means, stds), np.add(means, stds), alpha=.2, color='green')

plt.ylim(1.4, 1.6)
plt.xlim(0, 40000)

plt.xlabel('# Samples')
plt.ylabel('Surface Area')
plt.legend()
plt.show()
