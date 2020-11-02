# Names and UvAIDs:
# Matthijs de Wit, 10628258
# Menno Bruin, 11675225

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from tqdm import tqdm

rng = np.random.default_rng()


@njit
def mandelbrot(c, max_iterations):
    z = 0
    n = 0
    while n < max_iterations and abs(z) <= 2:
        z = z**2 + c
        n += 1
    return n


def random_sampling_method(num_samples, real_bounds, imag_bounds, max_iterations, plot_figures):
    print("running random sampling method...")

    # for plotting the set as figure
    x = []
    y = []
    colors = []

    samples_in_set = 0
    for _ in tqdm(range(num_samples)):
        c = complex(rng.uniform(*real_bounds), rng.uniform(*imag_bounds))
        n = mandelbrot(c, max_iterations)
        if n == max_iterations:
            samples_in_set += 1

        if plot_figures:
            x.append(c.real)
            y.append(c.imag)

            # give each sample a color dependent on n
            h = np.log(n)
            m = np.log(max_iterations)
            colors.append([0, m-h, 0])

    fraction_in_set = samples_in_set / num_samples
    total_area = (real_bounds[1] - real_bounds[0]) * (imag_bounds[1] - imag_bounds[0])
    print("estimated area using uniform random samples:", total_area * fraction_in_set)

    if plot_figures:
        print("plotting...")
        colors = np.array(colors)
        plt.scatter(x, y, s=1, c=colors/colors.max())
        plt.show()


if __name__ == "__main__":
    plot_figures = True

    max_iterations = 1000

    real_bounds = [-2, 0.5]
    imag_bounds = [-1.25, 1.25]

    random_sampling_method(int(4e5), real_bounds, imag_bounds, max_iterations, plot_figures)
