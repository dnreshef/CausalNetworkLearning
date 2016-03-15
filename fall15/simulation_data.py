import numpy as np

def parabola(sample_size, D):
    x = 2 * np.random.rand(sample_size, D) - 1 # [-1, 1)
    f_x = lambda x: [1 - (x[-1] ** 2) + 0.2 * np.random.standard_normal()]
    y = np.apply_along_axis(f_x, 1, x)
    #print("MEAN: ", np.mean(y))
    return x, y

def sine(sample_size, D):
    x = 4 * np.random.rand(sample_size, D) - 1 # [-1, 3)
    f_x = lambda x: [np.sin(x[-1] * np.pi) + 0.2 * np.random.standard_normal()]
    y = np.apply_along_axis(f_x, 1, x)
    return x, y

def line(sample_size, D):
    x = 4 * np.random.rand(sample_size, D) - 2 # [-2, 2)
    f_x = lambda x: [x[0] + 2 * x[1] + 0.1 * np.random.standard_normal()]
    y = np.apply_along_axis(f_x, 1, x)
    return x, y

def corrugated_curve(sample_size, D):
    x = 14 * np.random.rand(sample_size, D) - 7
    f_x = lambda x: [8 * np.sin(np.pi * x[-2]/2) - x[-1] ** 2 + 0.2 * np.random.standard_normal()]
    y = np.apply_along_axis(f_x, 1, x)
    return x, y
