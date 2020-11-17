# Names and UvAIDs:
# Matthijs de Wit, 10628258
# Menno Bruin, 11675225
#
# This file contains the different methods used for reducing variance in random sampling

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

    # for plotting the set as figure
    xs = []
    ys = []
    colors = []
    m = np.log(max_iterations)

    samples_in_set = 0
    for _ in tqdm(range(num_samples), disable=True):
        c = complex(rng.uniform(*real_bounds), rng.uniform(*imag_bounds))
        n = mandelbrot(c, max_iterations)
        if n == max_iterations:
            samples_in_set += 1

        if plot_figures:
            xs.append(c.real)
            ys.append(c.imag)

            # give each sample a color dependent on n
            h = np.log(n)
            r = max(0, m - 4 * abs(h - 0.25 * m))
            g = max(0, m - 4 * abs(h - 0.5 * m))
            b = max(0, m - 4 * abs(h - 0.75 * m))
            colors.append([r, g, b])

    fraction_in_set = samples_in_set / num_samples
    total_area = (real_bounds[1] - real_bounds[0]) * (imag_bounds[1] - imag_bounds[0])
    calculated_area = total_area * fraction_in_set

    if plot_figures:
        print("plotting...")
        colors = np.array(colors)
        plt.scatter(xs, ys, s=1, c=colors/colors.max())
        plt.show()

    return calculated_area


def alternative_antithetic_sampling_method(num_samples, real_bounds, imag_bounds, max_iterations, plot_figures):
    # doesn't work (yet)

    # for plotting the set as figure
    xs = []
    ys = []
    colors = []
    m = np.log(max_iterations)

    # constants
    real_width = real_bounds[1] - real_bounds[0]
    half_real_width = real_width / 2
    real_center = (real_bounds[0] + real_bounds[1]) / 2

    imag_width = imag_bounds[1] - imag_bounds[0]
    half_imag_width = real_width / 2
    imag_center = (imag_bounds[0] + imag_bounds[1]) / 2

    samples_in_set = 0
    for _ in tqdm(range(num_samples // 2), disable=True):
        # color = [rng.random(), rng.random(), rng.random()]
        c1 = complex(rng.uniform(*real_bounds), rng.uniform(*imag_bounds))

        x1 = c1.real - real_center
        y1 = c1.imag - imag_center
        if abs(x1) > abs(y1):  # assuming width real == width imag
            if x1 > 0:
                angle = np.arctan(y1 / x1)
                x2 = half_real_width - x1
                y2 = x2 * np.tan(angle)
                # plt.plot([c1.real, c1.real + x2], [c1.imag, c1.imag + y2], c=color)
                c2 = complex(real_center - x2, imag_center - y2)
            else:
                angle = np.arctan(y1 / x1)
                x2 = - half_real_width - x1
                y2 = x2 * np.tan(angle)
                # plt.plot([c1.real, c1.real + x2], [c1.imag, c1.imag + y2], c=color)
                c2 = complex(real_center - x2, imag_center - y2)
        else:
            if y1 > 0:
                angle = np.arctan(x1 / y1)
                y2 = half_imag_width - y1
                x2 = y2 * np.tan(angle)
                # plt.plot([c1.real, c1.real + x2], [c1.imag, c1.imag + y2], c=color)
                c2 = complex(real_center - x2, imag_center - y2)
            else:
                angle = np.arctan(x1 / y1)
                y2 = - half_imag_width - y1
                x2 = y2 * np.tan(angle)
                # plt.plot([c1.real, c1.real + x2], [c1.imag, c1.imag + y2], c=color)
                c2 = complex(real_center - x2, imag_center - y2)

        n1 = mandelbrot(c1, max_iterations)
        if n1 == max_iterations:
            samples_in_set += 1
        n2 = mandelbrot(c2, max_iterations)
        if n2 == max_iterations:
            samples_in_set += 1

        if plot_figures:
            xs.append(c1.real)
            ys.append(c1.imag)
            xs.append(c2.real)
            ys.append(c2.imag)

            # give each sample a color dependent on n
            h = np.log(n1)
            r = max(0, m - 4 * abs(h - 0.25 * m))
            g = max(0, m - 4 * abs(h - 0.5 * m))
            b = max(0, m - 4 * abs(h - 0.75 * m))
            colors.append([r, g, b])

            h = np.log(n2)
            r = max(0, m - 4 * abs(h - 0.25 * m))
            g = max(0, m - 4 * abs(h - 0.5 * m))
            b = max(0, m - 4 * abs(h - 0.75 * m))
            colors.append([r, g, b])

            # colors.append(color)
            # colors.append(color)

    fraction_in_set = samples_in_set / num_samples
    total_area = (real_bounds[1] - real_bounds[0]) * (imag_bounds[1] - imag_bounds[0])
    calculated_area = total_area * fraction_in_set

    if plot_figures:
        print("plotting...")
        colors = np.array(colors)
        plt.scatter(xs, ys, s=1, c=colors/colors.max())
        # plt.plot([-2, 0.5, 0.5, -2, -2], [1.25, 1.25, -1.25, -1.25, 1.25], c='black', lw=2)
        # plt.scatter([real_center], [imag_center], marker='+', s=20, c='black')
        plt.tight_layout()
        plt.show()

    return calculated_area


def antithetic_sampling_method(num_samples, real_bounds, imag_bounds, max_iterations, plot_figures):

    # for plotting the set as figure
    xs = []
    ys = []
    colors = []
    m = np.log(max_iterations)

    # constants
    real_width = real_bounds[1] - real_bounds[0]
    half_real_width = real_width / 2

    imag_width = imag_bounds[1] - imag_bounds[0]
    half_imag_width = real_width / 2
    imag_center = (imag_bounds[0] + imag_bounds[1]) / 2

    samples_in_set = 0
    for _ in tqdm(range(num_samples // 2), disable=True):
        # color = [rng.random(), rng.random(), rng.random()]
        c1 = complex(rng.uniform(*real_bounds), rng.uniform(imag_bounds[0], 0))

        if c1.imag < imag_center:
            c2 = complex(c1.real, c1.imag + half_imag_width)
        else:
            c2 = complex(c1.real, c1.imag - half_imag_width)

        n1 = mandelbrot(c1, max_iterations)
        if n1 == max_iterations:
            samples_in_set += 1
        n2 = mandelbrot(c2, max_iterations)
        if n2 == max_iterations:
            samples_in_set += 1

        if plot_figures:
            xs.append(c1.real)
            ys.append(c1.imag)
            xs.append(c2.real)
            ys.append(c2.imag)

            # give each sample a color dependent on n
            h = np.log(n1)
            r = max(0, m - 4 * abs(h - 0.25 * m))
            g = max(0, m - 4 * abs(h - 0.5 * m))
            b = max(0, m - 4 * abs(h - 0.75 * m))
            colors.append([r, g, b])

            h = np.log(n2)
            r = max(0, m - 4 * abs(h - 0.25 * m))
            g = max(0, m - 4 * abs(h - 0.5 * m))
            b = max(0, m - 4 * abs(h - 0.75 * m))
            colors.append([r, g, b])

            # colors.append(color)
            # colors.append(color)

    fraction_in_set = samples_in_set / num_samples
    total_area = (real_bounds[1] - real_bounds[0]) * (imag_bounds[1] - imag_bounds[0])
    calculated_area = total_area * fraction_in_set

    if plot_figures:
        print("plotting...")
        colors = np.array(colors)
        plt.scatter(xs, ys, s=1, c=colors/colors.max())
        # plt.plot([-2, 0.5, 0.5, -2, -2], [1.25, 1.25, -1.25, -1.25, 1.25], c='black', lw=2)
        # plt.plot([-2, 0.5], [0,0], '--', c='black', lw=2)
        plt.tight_layout()
        plt.show()

    return calculated_area


def control_variate_pilot(num_samples, real_bounds, imag_bounds, max_iterations, circles):

    samples_in_set = []
    samples_in_circles = []
    for _ in tqdm(range(num_samples)):
        c = complex(rng.uniform(*real_bounds), rng.uniform(*imag_bounds))

        n = mandelbrot(c, max_iterations)
        if n == max_iterations:
            samples_in_set.append(1)
        else:
            samples_in_set.append(0)

        for circle in circles:
            if np.sqrt((c.real - circle["c"][0]) ** 2 + (c.imag - circle["c"][1]) ** 2) < circle["r"]:
                samples_in_circles.append(1)
                break
        else:
            samples_in_circles.append(0)

    s = min(len(samples_in_set), len(samples_in_circles))
    correlation = np.cov(samples_in_set[:s], samples_in_circles[:s])[0][1]
    variance = np.var(samples_in_circles)
    cv = -correlation/variance
    # print(correlation, variance, cv)
    return cv


def control_variate_sampling_method(num_samples, real_bounds, imag_bounds, max_iterations, circles, cv, plot_figures):

    # for plotting the set as figure
    xs = []
    ys = []
    colors = []
    m = np.log(max_iterations)

    # assuming circles don't overlap!
    total_area_circles = sum([np.pi * (circle["r"]**2) for circle in circles])

    samples_in_set = 0
    samples_in_circles = 0
    for _ in tqdm(range(num_samples), disable=True):
        c = complex(rng.uniform(*real_bounds), rng.uniform(*imag_bounds))

        n = mandelbrot(c, max_iterations)
        if n == max_iterations:
            samples_in_set += 1

        for circle in circles:
            if np.sqrt((c.real - circle["c"][0]) ** 2 + (c.imag - circle["c"][1]) ** 2) < circle["r"]:
                samples_in_circles += 1
                break

        if plot_figures:
            xs.append(c.real)
            ys.append(c.imag)

            # give each sample a color dependent on n
            h = np.log(n)
            r = max(0, m - 4 * abs(h - 0.25 * m))
            g = max(0, m - 4 * abs(h - 0.5 * m))
            b = max(0, m - 4 * abs(h - 0.75 * m))
            colors.append([r, g, b])

    fraction_in_set = samples_in_set / num_samples
    fraction_in_circles = samples_in_circles / num_samples

    total_area = (real_bounds[1] - real_bounds[0]) * (imag_bounds[1] - imag_bounds[0])

    x = total_area * fraction_in_set
    y = total_area * fraction_in_circles
    y_m = total_area_circles

    calculated_area = x + cv * (y - y_m)

    if plot_figures:
        print("plotting...")
        fig = plt.gcf()
        ax = fig.gca()

        colors = np.array(colors)
        plt.scatter(xs, ys, s=1, c=colors/colors.max())

        for circle in circles:
            circle_patch = plt.Circle(circle["c"], circle["r"], color='b', fill=False)
            ax.add_artist(circle_patch)

        plt.tight_layout()
        plt.show()

    return calculated_area


if __name__ == "__main__":
    plot_figures = True
    calculate_cv = False

    max_iterations = 1000
    num_samples = int(1e5)

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

    if calculate_cv:
        cv = control_variate_pilot(int(1e6), real_bounds, imag_bounds, 1000, cv_circles)
        print(cv)
    else:
        # previous estimate
        cv = -0.9248455094960547

    a = random_sampling_method(num_samples, real_bounds, imag_bounds, max_iterations, plot_figures)
    print("estimated area (random):", a)

    a = antithetic_sampling_method(num_samples, real_bounds, imag_bounds, max_iterations, plot_figures)
    print("estimated area (antithetic):", a)

    a = control_variate_sampling_method(num_samples, real_bounds, imag_bounds, max_iterations, cv_circles, cv, plot_figures)
    print("estimated area (control variate):", a)
