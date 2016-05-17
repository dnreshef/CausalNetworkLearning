#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

def plot_acquisition(acquisition_func, granularity):
    """
    Plots a function of two variables, from -100*granularity to 100*granularity
    in both x and y axes.

    Args:
        acquisition_func (function): Accepts two floats and outputs a float
        granularity (float): Level of zoom.

    Returns: None
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(-100* granularity, 101* granularity, granularity)
    Y = np.arange(-100* granularity, 101* granularity, granularity)
    X, Y = np.meshgrid(X, Y)
    F = np.vectorize(acquisition_func)
    Z = F(X, Y)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def plot_2D(predict, D, rds, intervals):
    """
    Plots a predict function along two dimensions. Shows 5th and 95th quantiles.

    Args:
        predict (function): predict function of a GPy model
        D (int): total dimensions
        rds (tuple(int)): dimensions to be plotted
        intervals (tuple(tuple(int))): of the form ((a, b), (c, d)) so that the
            the function is plotted on [a, b] x [c, d].

    Returns: None
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.linspace(intervals[0][0], intervals[0][1], 200)
    Y = np.linspace(intervals[1][0], intervals[1][1], 200)
    X, Y = np.meshgrid(X, Y)
    z = np.zeros((X.size, D))
    z[:,rds] = np.hstack((X.reshape((X.size, 1)), Y.reshape((Y.size, 1))))
    zmu, zs2 = predict(z)
    Z = np.reshape(zmu - 1.645 * np.sqrt(zs2), X.shape)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def plot_1D(predict, D, rd, interval):
    """
    Plots a predict function along one dimension. Shows 5th and 95th quantiles.

    Args:
        predict (function): predict function of a GPy model
        D (int): total dimensions
        rd (int): dimension to be plotted
        interval (tuple(int)): interval [a, b] on which to plot

    Returns: None
    """
    x = np.linspace(interval[0], interval[1], 200)
    X = np.zeros((200, D))
    X[:,rd] = x
    print(X.shape)
    ymu, ys2 = predict(X)
    ymu = ymu.flatten()
    ys = np.sqrt(ys2.flatten())
    plt.plot(x, ymu, ls='-', marker='None')
    plt.fill_between(x, ymu - 1.645 * ys, ymu + 1.645 * ys, facecolor=[0.7539, 0.89453125, 0.62890625, 1.0], linewidths=0)
    plt.show()

def plot_data(x_data, y_data, dimension):
    """
    Plots the data along dimension.

    Args:
        x_data (numpy.ndarray): n by D array of data
        y_data (numpy.ndarray): 1d array of length n
        dimension (int): dimensio to be plotted

    Returns: None
    """
    plt.figure()
    plt.plot(x_data[:,-1].flatten(), y_data, ls='None', marker='+')
    plt.show()

def main():
    prefix = 'plots/plane_ss'
    no_smoothing = np.load(prefix + '.npy')[:6,:]
    smoothing = np.load(prefix + '_smoothing.npy')[:6,:]
    restarts = np.load(prefix + '_restarts.npy')[:6,:]
    initial_vals = np.load(prefix + '_initial_vals.npy')
    true_opts = np.load(prefix + '_true_opts.npy')

    var_array = np.array([20, 30, 40, 50, 75, 100])

    xlim = (var_array[0] - 0.03 * var_array[-1], var_array[-1] * 1.03)
    plt.figure()
    plt.title('Objective gain vs sample size')
    plt.xlabel('sample_size')
    plt.ylabel('objective gain')
    plt.xlim(xlim)
    for name in ['no_smoothing', 'smoothing', 'restarts']:
        f_values = eval(name)
        avg_y_gain = np.apply_along_axis(lambda x: sum(x - initial_vals) / len(x), 1, f_values)
        percentile_5_y_gain = np.apply_along_axis(lambda x: np.percentile(x - initial_vals, 5), 1, f_values)
        plt.plot(var_array, avg_y_gain, ls='-', label=name+' mean')
        plt.plot(var_array, percentile_5_y_gain, ls='-', label=name+' 5th percentile')
    iv = initial_vals
    to = true_opts
    horiz_lines = [np.average(to - iv), np.percentile(to - iv, 5)]
    plt.hlines(horiz_lines, xlim[0], xlim[1], colors=['k', 'b'], linestyles='dashed')
    plt.legend(loc=0,prop={'size':10})
    plt.savefig(prefix + '_aggregate')

if __name__ == "__main__":
    main()
