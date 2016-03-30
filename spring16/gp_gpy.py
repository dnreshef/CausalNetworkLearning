from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.stats import multivariate_normal as mvn
from simulation import *
import GPy
import GPyOpt
import math
import sys

def find_sparse_intervention(objective_and_grad, test_point, intervention_dim=1):
    """
    Finds an intervention constrained to differ from the test point in at most
    intervention_dim dimensions.

    Args:
        objective_and_grad (function): Accepts 1D array of length
            test_point.size, outputs objective, gradient (float, numpy.ndarray)
            The objective function is maximized.
        test_point (numpy.ndarray): 1D array representing the initial guess
        intervention_dim (int): Max number of dimensions in which intervention
            can differ from test_point

    Returns:
        numpy.ndarray: 1D array of length test_point.size, the optimal intervention
    """
    negate_tuple = lambda tup: tuple(-1 * x for x in tup)
    def restricted_objective(dims):
        def fun(x):
            guess = np.copy(test_point)
            guess[dims] = x
            return negate_tuple(objective_and_grad(guess))
        return fun

    # TODO: Support regularization
    def regularized_objective(l):
        return lambda x: negate_tuple(objective_and_grad(x))# + l * np.linalg.norm(x - test_point, ord=1)
    reg_constants = [0]#np.power(10.0, np.arange(-7, 7)).tolist()

    # Identify dimensions to be optimized
    log_search = True
    last_dim_diff = None
    x_opt = None
    while len(reg_constants) > 0:
        l = reg_constants.pop()
        x_opt = minimize(regularized_objective(l),
                         test_point,
                         method='BFGS',
                         jac=True,
                         options={'disp':False}).x
        if np.count_nonzero(x_opt != test_point) > intervention_dim:
            assert last_dim_diff is not None
            if log_search:
                reg_constants = (np.arange(2, 10) * l).tolist()
                log_search = False
            else: 
                break
        else:
            last_dim_diff = x_opt != test_point

    # If dimensions found, now optimize without regularization
    if np.count_nonzero(last_dim_diff) == 0:
        print("I couldn't optimize the test point at all.")
        return test_point
    else: 
        print("Optimizing these dimensions:",last_dim_diff)
        restricted_opt = minimize(restricted_objective(last_dim_diff),
                                  test_point[last_dim_diff],
                                  jac=True,
                                  options={'disp':False}).x
        ret = np.copy(test_point)
        ret[last_dim_diff] = restricted_opt
        return ret

def gpyopt_intervention(objective, test_point):
    """
    Deprecated. Use find_sparse_intervention instead.
    """
    D = len(test_point)
    def objective_wrapper(x):
        x = x.reshape((x.size/D, D))
        n = x.shape[0]
        return np.apply_along_axis(lambda x: -objective(x), 1, x).reshape(n, 1)
    opt = GPyOpt.methods.BayesianOptimization(
        f=objective_wrapper, bounds=[(-2,2) for i in xrange(test_point.size)])
    opt.run_optimization(20)
    opt.plot_acquisition()
    opt.plot_convergence()
    return opt.x_opt

def validate_gpy():
    D = 5
    n_range = range(610, 9, -200)
    n_heldout = 1000
    fixed_x, fixed_y, z = line[0](n_heldout, D)

    for i in xrange(len(n_range)):
        n = n_range[i]
        print("Sample size", n)
        training_x, training_y, z = line[0](n, D)
        y = training_y.flatten()
        kernel = GPy.kern.RBF(D, ARD=True)
        model = GPy.models.GPRegression(training_x, training_y, kernel)
        model.optimize()
        kernel.lengthscale = 2 * original_lengthscale
        y_pred, sigma2_pred = model.predict(fixed_x)

        model.plot(fixed_inputs=[(i,0) for i in xrange(2, D)])
        plt.show()
        print("MSE: ", np.mean( (fixed_y.flatten()-y_pred.flatten()) ** 2))
        print("Likelihood function:", mvn.logpdf(
            fixed_y.flatten(), y_pred.flatten(), np.diag(sigma2_pred.flatten())))
        print(model)
        print(kernel.lengthscale)

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

def plot_2D(predict, D, granularity):
    # Only works with the first two dims being relevant
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(-100* granularity, 101* granularity, granularity)
    Y = np.arange(-100* granularity, 101* granularity, granularity)
    X, Y = np.meshgrid(X, Y)
    z = np.hstack((X.reshape((X.size, 1)), Y.reshape((Y.size, 1)), np.zeros((X.size, D-2))))
    zmu, zs2 = predict(z)
    Z = np.reshape(zmu - 1.645 * np.sqrt(zs2), X.shape)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def plot_1D(predict, D):
    # Only works for range [-1, 1] with the last dimension being relevant
    x = np.arange(-1, 1 - 1e-14, 1e-2)
    print(x.size)
    X = np.hstack((np.zeros((200, D-1)), x.reshape((200, 1))))
    ymu, ys2 = predict(X)
    ymu = ymu.flatten()
    ys = np.sqrt(ys2.flatten())
    plt.plot(x, ymu-1.645*ys)
    #plt.plot(x, ymu, ls='None', marker='+')
    #plt.fill_between(x, ymu - 1.645 * ys, ymu + 1.645 * ys, facecolor=[0.7539, 0.89453125, 0.62890625, 1.0], linewidths=0)
    plt.show()

def run(training_x, training_y, test_point, plot_dims=None, intervention_dim=1):
    """
    Runs the intervention optimization routine. A model is trained from
    training_x and training_y, and is used to find a sparse intervention on
    test_point.

    Args:
        training_x (numpy.ndarray): n x m array with training data, where there
            are n data points of m dimensions
        training_y (numpy.ndarray): m x 1 array with the y-value of the
            training data
        test_point (numpy.ndarray): The test point, a 1D array of length m
        plot_dims (numpy.ndarray): If plot_dims is specified, this must
            be a 1D array with the indices of the dimensions to be plotted.
        intervention_dim (int): Max number of dimensions in which the
            intervention can differ from test_point

    Returns:
        numpy.ndarray: 1D array representing the optimal intervention.
    """
    FIVE_PERCENTILE_ZSCORE = -1.645
    D = training_x.shape[1]
    print("=" * 60)
    print("Running with sample size {ss} in a {d}-D space".format(
        ss=training_x.shape[0], d=D))
    if plot_dims is not None:
        correct_dims = set(dim_index if dim_index >= 0 else D + dim_index\
            for dim_index in plot_dims)
        gpy_plot_fixed_inputs = [(i, 0) for i in xrange(D) if i not in correct_dims]

    x, y = training_x, training_y
    y = y.flatten()

    kernel = GPy.kern.RBF(D, ARD=True)
    model = GPy.models.GPRegression(training_x, training_y, kernel)

    model.optimize()

    print("Optimized gp parameters. Now finding optimal intervention.")
    print("Test point:", test_point)
    print()

    def objective_and_grad(x_star):
        x = x_star.reshape((1, x_star.size))
        y_mu, y_cov = model.predict(x)
        y_mu = y_mu[0][0]
        y_var = y_cov[0][0]
        dmu_dX, dv_dX = model.predictive_gradients(x)
        dmu_dX = dmu_dX.reshape((dmu_dX.size,))
        dv_dX = dv_dX[0]
        ret = y_mu + FIVE_PERCENTILE_ZSCORE * np.sqrt(y_var)
        grad = dmu_dX - FIVE_PERCENTILE_ZSCORE * (0.5 / np.sqrt(y_var)) * dv_dX
        return ret, grad

    def acq(x, y):
        fmu, fv = model._raw_predict(np.array([[x, y]]))
        ret, = model.likelihood.predictive_quantiles(fmu, fv, (5,))
        return ret[0][0]

    #plot_2D(model.predict, D, 1e-8)

    # Smoothing
    orig_lengthscale = kernel.lengthscale.copy()
    orig_variance = kernel.variance.copy()
    current_test_point = test_point
    model.kern.lengthscale.fix()
    for i in xrange(0, -1, -1):
        #print("Lengthscales multiplied by {}".format(np.exp(i)))
        #model.kern.lengthscale = np.exp(i) * orig_lengthscale

        # Recompute variance and noise scale
        #model.optimize()

        current_test_point = find_sparse_intervention(
            objective_and_grad, current_test_point, intervention_dim=D)
        print("Test point reinitialized to", current_test_point)
        #print("Acquisition:", objective_and_grad(current_test_point)[0])
        mean, var = model.predict(current_test_point.reshape((1, current_test_point.size)))
        #print("Mean:", mean)
        #print("Var:", var)

        #plot_1D(model.predict, D)

        if plot_dims is not None:
            model.plot(fixed_inputs=gpy_plot_fixed_inputs, plot_limits=[-1e-8, 1e-8])
            plt.show()

        print()
        model.kern.variance = orig_variance

    print()
    return current_test_point

def analyze(sample_size, dimension, repeat, filename, simulation):
    simulation_func, simulation_eval, correct_dims = simulation

    avg_x_diff = []
    std_devs_x = []
    avg_y_diff = []
    std_devs_y = []
    correct_dim_found_count = 0

    independent_var = None
    var_array = None
    var_to_tuple = None

    if isinstance(sample_size, int):
        independent_var = 'dimension'
        var_array = dimension
        var_to_tuple = lambda dimension: (sample_size, dimension)
    elif isinstance(dimension, int):
        independent_var = 'sample size'
        var_array = sample_size
        var_to_tuple = lambda sample_size: (sample_size, dimension)
    else:
        print('Analyze called with incorrect syntax.')
        sys.exit(2)

    for var in var_array:
        devs_x = []
        devs_y = []
        for i in xrange(repeat):
            test_point = simulation_func(1,
                dimension if independent_var == 'sample size' else var)[0][0]
            print(test_point)
            training_x, training_y = simulation_func(*var_to_tuple(var))
            x_opt = run(training_x, training_y, test_point, intervention_dim=100) # Hack to avoid regularization
            if np.all((x_opt != test_point)[correct_dims]):
                correct_dim_found_count += 1
            x_diff, y_diff = simulation_eval(x_opt)
            devs_x.append(x_diff)
            devs_y.append(y_diff)
        avg_x_diff.append(sum(devs_x) / repeat)
        avg_y_diff.append(sum(devs_y) / repeat)
        std_devs_x.append(np.std(devs_x))
        std_devs_y.append(np.std(devs_y))

    print("Correct intervention dimension was identified in {} out of {} runs."\
          .format(correct_dim_found_count, len(var_array) * repeat))
    xlim = (0, math.ceil(var_array[-1] * 1.03))
    plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')

    plt.subplot(121)
    plt.title('intervention distance vs {}'.format(independent_var))
    plt.xlabel(independent_var)
    plt.ylabel('intervention distance')
    plt.xlim(xlim)
    plt.plot(var_array, avg_x_diff, ls='-', marker='o')
    plt.errorbar(var_array, avg_x_diff, yerr=np.array(std_devs_x))

    plt.subplot(122)
    plt.title('objective difference vs {}'.format(independent_var))
    plt.xlabel(independent_var)
    plt.ylabel('objective difference')
    plt.xlim(xlim)
    plt.plot(var_array, avg_y_diff, ls='-', marker='o')
    plt.errorbar(var_array, avg_y_diff, yerr=np.array(std_devs_y))

    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def main():
    sample_size_array = np.array([10, 20, 30, 40, 50, 75, 100, 150, 200, 300])
    dimension = 10
    sample_size = 100
    dimensions_array = np.arange(2, 16)
    repeat = 2
    for trial in ['parabola', 'paraboloid', 'sine', 'line', 'plane', 'corrugated_curve']:
        exec "analyze(sample_size_array, dimension, repeat, 'plots/{trial}_ss.png', {trial})".format(trial=trial)
        exec "analyze(sample_size, dimensions_array, repeat, 'plots/{trial}_d.png', {trial})".format(trial=trial)

if __name__ == "__main__":
    main()

