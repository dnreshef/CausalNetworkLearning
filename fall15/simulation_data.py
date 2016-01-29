import numpy as np

def parabola(sample_size=100, D=2):
    x = 2 * np.random.rand(sample_size, D) - 1 # D-dimensional, uniform in [-1, 1)
    f_x = lambda x: [1 - x[-1] ** 2 + 0.1 * np.random.standard_normal()]
    y = np.apply_along_axis(f_x, 1, x)
    z = 2 * np.random.rand(sample_size, D) - 1 # D-dimensional, uniform in [-1, 1)
    return x, y, z # train x, train y, test x