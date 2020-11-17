# Names and UvAIDs:
# Matthijs de Wit, 10628258
# Menno Bruin, 11675225

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from tqdm import tqdm
import pickle as pk

rng = np.random.default_rng()


@njit
def mandelbrot(c, max_iterations):
    z = 0
    n = 0
    while n < max_iterations and abs(z) <= 2:
        z = z ** 2 + c
        n += 1
    return n


def orthogonal_lhs(n, x_bound: list, y_bound: list):
    """
    Orthogonal Latin Hypercube sampling method

    :param n: number of subspaces
    :param x_bound: boundaries of x -> [xmin, xmax]
    :param y_bound: boundaries of y -> [ymin, ymax]
    :return: n^2 orthogonally distributed random samples on the bounded domain
    """
    x_min, x_max = x_bound[0], x_bound[1]
    dx = (x_max - x_min)
    y_min, y_max = y_bound[0], y_bound[1]
    dy = (y_max - y_min)

    samples = n * n
    x_scale = dx / samples
    y_scale = dy / samples

    x, y = np.arange(0, samples).reshape((n, n)), np.arange(0, samples).reshape((n, n))
    for i in range(n):
        np.random.shuffle(x[i])
        np.random.shuffle(y[i])

    x = x_min + x_scale * np.add(x, np.random.rand(n, n))
    y = y_min + y_scale * np.add(y, np.random.rand(n, n)).T

    points = np.vstack((x.flatten(), y.flatten())).T
    return points[:, 0] + 1j * points[:, 1]


def lhs(n_samples, x_bound: list, y_bound: list):
    """
    Latin Hypercube sampling

    :param n_samples: number of samples
    :param x_bound: boundaries of x -> [xmin, xmax]
    :param y_bound: boundaries of y -> [ymin, ymax]
    :return: n_samples random samples distributed on a Latin Hypercube
    """
    x_min, x_max = x_bound[0], x_bound[1]
    dx = (x_max - x_min) / n_samples
    y_min, y_max = y_bound[0], y_bound[1]
    dy = (y_max - y_min) / n_samples

    lower_lim_x = np.linspace(x_min, x_max, n_samples)
    upper_lim_x = np.linspace(x_min + dx, x_max + dx, n_samples)
    lower_lim_y = np.linspace(y_min, y_max, n_samples)
    upper_lim_y = np.linspace(y_min + dy, y_max + dy, n_samples)

    x_coords = np.random.uniform(lower_lim_x, upper_lim_x, size=n_samples).T
    y_coords = np.random.uniform(lower_lim_y, upper_lim_y, size=n_samples).T

    points = np.vstack((x_coords, y_coords)).T
    np.random.shuffle(points[:, 1])
    points = points[:, 0] + 1j * points[:, 1]

    return points


def calculate_mandelbrot_area(num_samples, num_subspaces, real_bounds, imag_bounds, max_iterations, plot_figures, method=None):
    """
    Calculation of the area of the Mandelbrot set

    :param num_samples: number of samples
    :param num_subspaces: numbers of subspaces
    :param real_bounds: boundaries of x -> [xmin, xmax]
    :param imag_bounds: boundaries of y -> [ymin, ymax]
    :param max_iterations: maximum number of iterations
    :param plot_figures: True or False
    :param method: sampling method to use ['orthogonal_lhs', 'lhs', None]
    :return: calculated area, number of samples
    """
    x = []
    y = []
    colors = []
    if method == 'orthogonal_lhs':
        points = orthogonal_lhs(num_subspaces, x_bound=real_bounds, y_bound=imag_bounds)
    elif method == 'lhs':
        points = lhs(num_samples, x_bound=real_bounds, y_bound=imag_bounds)
    else:
        points = [complex(rng.uniform(*real_bounds), rng.uniform(*imag_bounds)) for _ in range(num_samples)]

    samples_in_set = 0
    for i in range(num_samples):
        c = points[i]
        n = mandelbrot(c, max_iterations)
        if n == max_iterations:
            samples_in_set += 1

        if plot_figures and n == max_iterations:
            x.append(c.real)
            y.append(c.imag)

            # give each sample a color dependent on n
            h = np.log(n)
            m = np.log(max_iterations)
            r = max(0, m - 4 * abs(h - 0.25 * m))
            g = max(0, m - 4 * abs(h - 0.5 * m))
            b = max(0, m - 4 * abs(h - 0.75 * m))
            colors.append([r, g, b])

    fraction_in_set = samples_in_set / num_samples
    total_area = (real_bounds[1] - real_bounds[0]) * (imag_bounds[1] - imag_bounds[0])

    if plot_figures:
        print("estimated area using uniform random samples:", total_area * fraction_in_set)
        print("plotting...")
        colors = np.array(colors)
        plt.scatter(x, y, s=10, c=colors/colors.max())
        plt.show()

    return total_area * fraction_in_set, num_samples


if __name__ == "__main__":
    plot_figures = False

    max_iterations = 1e4
    num_subspaces = 1e6
    max_samples = 1e6

    real_bounds = [-2, 0.5]
    imag_bounds = [-1.25, 1.25]

    samples, means, stds = [], [], []
    n = 25

    for num_subspaces in tqdm(range(5, 205, 5)):
        max_samples = num_subspaces * num_subspaces
        surfaces = []
        for i in range(0, n):
            sampling = calculate_mandelbrot_area(int(max_samples), num_subspaces, real_bounds, imag_bounds, max_iterations, plot_figures=False)
            surface = sampling[0]
            surfaces.append(surface)

        samples.append(max_samples)
        means.append(np.mean(surfaces))
        stds.append(np.std(surfaces))

    plt.plot(samples, means, '.')
    plt.fill_between(samples, np.subtract(means, stds), np.add(means, stds), alpha=.2)
    plt.show()

    with open('random_sampling_data', 'wb+') as f:
        pk.dump({"samples": samples, "means": means, "stds": stds}, f)

