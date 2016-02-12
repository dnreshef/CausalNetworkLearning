import numpy as np

def parabola(sample_size, D):
    x = 2 * np.random.rand(sample_size, D) - 1 # D-dimensional, uniform in [-1, 1)
    f_x = lambda x: [1 - (x[-1] ** 2) + 0.2 * np.random.standard_normal()]
    y = np.apply_along_axis(f_x, 1, x)
    plot_x = np.hstack((2 * np.random.rand(100, D-1) - 1, np.linspace(-1.2, 1.2, 100).reshape((100, 1))))
    return x, y, plot_x

def corrugated_curve(sample_size, D):
    x = 14 * np.random.rand(sample_size, D) - 7
    f_x = lambda x: [8 * np.sin(np.pi * x[-2]/2) - x[-1] ** 2 + 0.2 * np.random.standard_normal()]
    y = np.apply_along_axis(f_x, 1, x)
    return x, y, None
