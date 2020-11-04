import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from assignment1 import random_sampling_method

if __name__ == "__main__":
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')

    s_list = np.logspace(2, 6, 10)
    i_list = np.logspace(2, 5, 10)

    X, Y = np.meshgrid(s_list, i_list)

    real_bounds = [-2, 0.5]
    imag_bounds = [-1.25, 1.25]

    results = np.zeros((len(s_list), len(i_list)))
    for x, s in enumerate(s_list):
        for y, i in enumerate(i_list):
            individual_results = []
            for _ in range(5):
                a = random_sampling_method(int(s), real_bounds, imag_bounds, int(i), False)
                individual_results.append(a)
            
            results[x][y] = np.std(individual_results)

    print("X:", X.shape)
    print("Y:", Y.shape)
    print("X:", results.shape)

    ax.plot_surface(X, Y, results, cmap=cm.coolwarm)
    ax.set_xlabel('samples')
    ax.set_ylabel('iterations')
    ax.set_zlabel('standard deviation')

    plt.show()
