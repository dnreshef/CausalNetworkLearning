#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from simulation import *
from plotting import *
from validation import check_gradient
import GPy
import sys

def find_sparse_intervention(objective_and_grad, test_point, intervention_dim=None):
    """
    Finds an intervention constrained to differ from the test point in at most
    intervention_dim dimensions.

    Args:
        objective_and_grad (function): Accepts 1D array of length
            test_point.size, outputs objective, gradient (float, numpy.ndarray)
            The objective function is maximized.
        test_point (numpy.ndarray): 1D array representing the initial guess
        intervention_dim (int): Max number of dimensions in which intervention
            can differ from test_point. If none, optimization is unconstrained.

    Returns:
        numpy.ndarray: 1D array of length test_point.size, the optimal intervention
    """
    def restricted_objective(dims):
        def fun(x):
            guess = np.copy(test_point)
            guess[dims] = x
            obj, grad = objective_and_grad(guess)
            return -obj, -grad[dims]
        return fun

    def negative_objective(x):
        o, g = objective_and_grad(x)
        return -o, -g

    if intervention_dim is None:
        x_opt = minimize(negative_objective,
                         test_point,
                         method='BFGS',
                         jac=True,
                         options={'disp':False}).x
        return x_opt
    #reg_constants = np.power(10.0, np.arange(-7, 7)).tolist()
    l = 100.0

    last_dim_diff = None
    x_opt = None
    # Initialize upper bound
    while True:
        print("Trying l=", l)
        x_opt = gd_soft_threshold(negative_objective, test_point, l)
        if np.count_nonzero(x_opt != test_point) > intervention_dim:
            l *= 100
        else:
            last_dim_diff = x_opt != test_point
            break

    # Binary search
    ub = l
    lb = 0
    iteration = 0
    while True:
        if iteration >= 20:
            break
        l = (lb + ub) / 2
        print("Trying l=", l)
        x_opt = gd_soft_threshold(negative_objective, test_point, l)
        sparsity = np.count_nonzero(x_opt != test_point)
        if sparsity > intervention_dim:
            lb = l
        elif sparsity < intervention_dim:
            last_dim_diff = x_opt != test_point
            ub = l
        else:
            last_dim_diff = x_opt != test_point
            break
        iteration += 1

    # If dimensions found, now optimize without regularization
    if np.count_nonzero(last_dim_diff) == 0:
        print("I couldn't optimize the test point at all.")
        return test_point
    else: 
        print("Optimizing these dimensions:", last_dim_diff)
        restricted_opt = minimize(restricted_objective(last_dim_diff),
                                  test_point[last_dim_diff],
                                  jac=True,
                                  options={'disp':False}).x
        ret = np.copy(test_point)
        ret[last_dim_diff] = restricted_opt
        return ret

def gd_soft_threshold(objective_and_grad, initial, l):
    prev = float('inf')
    point = np.zeros(initial.size)
    iteration = 0
    prev_grad = 0
    eta = 1.
    beta = 0.5
    while True:
        o, grad = objective_and_grad(point + initial)
        o += l * np.linalg.norm(point, ord=1)
        if prev - o < 1e-9 and iteration >= 10:
            return point + initial
        elif prev - o < (eta/2) * np.linalg.norm(prev_grad)**2:
            eta = eta * beta
        prev = o
        prev_grad = grad
        update = grad * eta
        point = point - update
        point = np.sign(point) * np.maximum(np.abs(point) - l * eta, 0)
        iteration += 1

def run_population(training_x, training_y):
    FIVE_PERCENTILE_ZSCORE = -1.645
    D = training_x.shape[1]
    print("=" * 60)
    print("Running with sample size {ss} in a {d}-D space".format(
        ss=training_x.shape[0], d=D))
    x, y = training_x, training_y
    y = y.flatten()

    kernel = GPy.kern.RBF(D, ARD=True)
    model = GPy.models.GPRegression(training_x, training_y, kernel)

    model.optimize()

    def population_objective_and_grad(x_diff):
        new_pop = x + np.tile(x_diff, (len(x), 1))
        y_mu, y_var = model.predict(new_pop)
        ret = np.sum(y_mu + FIVE_PERCENTILE_ZSCORE * np.sqrt(y_var))
        dmu_dX, dv_dX = model.predictive_gradients(new_pop)
        dmu_dX = dmu_dX[:,:,0]
        grad = np.sum(dmu_dX + FIVE_PERCENTILE_ZSCORE * (0.5 / np.sqrt(y_var)) * dv_dX, axis=0) 
        return ret/len(x), grad/len(x)

    return find_sparse_intervention(population_objective_and_grad, np.zeros(D))

def run(training_x, training_y, test_point, plot_dims=None, intervention_dim=None):
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
            intervention can differ from test_point. If None, optimization is
            unconstrained.

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

    def objective_and_grad(x_star):
        x = x_star.reshape((1, x_star.size))
        y_mu, y_var = model.predict(x)
        y_mu = y_mu[0][0]
        y_var = y_var[0][0]
        dmu_dX, dv_dX = model.predictive_gradients(x)
        dmu_dX = dmu_dX.reshape((dmu_dX.size,))
        dv_dX = dv_dX[0]
        ret = y_mu + FIVE_PERCENTILE_ZSCORE * np.sqrt(y_var)
        grad = dmu_dX + FIVE_PERCENTILE_ZSCORE * (0.5 / np.sqrt(y_var)) * dv_dX
        return ret, grad

    def acq(x, y):
        fmu, fv = model._raw_predict(np.array([[x, y]]))
        ret, = model.likelihood.predictive_quantiles(fmu, fv, (5,))
        return ret[0][0]

    # Validate plane
    #check_gradient(lambda x: (model.predict(x.reshape((1, 10)))[1][0][0], model.predictive_gradients(x.reshape((1, 10)))[1].reshape((10,))), 10, 0)
    #check_gradient(objective_and_grad, 10, 0)
    #check_gradient(objective_and_grad, 10, 1)

    print("Optimized gp parameters. Now finding optimal intervention.")
    print("Test point:", test_point)
    print()

    # Smoothing
    orig_lengthscale = kernel.lengthscale.copy()
    orig_variance = kernel.variance.copy()
    current_test_point = test_point
    model.kern.lengthscale.fix()
    for i in (4, 2, 1):
        model.kern.lengthscale = i * orig_lengthscale
        # Recompute variance and noise scale
        model.optimize()

        opt = find_sparse_intervention(
            objective_and_grad, current_test_point, intervention_dim)
        print("Test point will be reinitialized to", opt)

        current_test_point = opt
        #print("Acquisition and gradient:", objective_and_grad(current_test_point))

        if plot_dims is not None:
            model.plot(fixed_inputs=gpy_plot_fixed_inputs)
            plt.show()

        print()
        model.kern.variance = orig_variance

    print()
    return current_test_point

def analyze(sample_size, dimension, noise, repeat, filename, simulation,
            population=False):
    simulation_func, simulation_eval, correct_dims = simulation

    avg_x_diff = []
    std_devs_x = []
    avg_y_diff = []
    std_devs_y = []
    correct_dim_found_count = 0

    independent_var = None
    var_array = None
    var_to_tuple = None

    if isinstance(dimension, np.ndarray):
        independent_var = 'dimension'
        var_array = dimension
        var_to_tuple = lambda dimension: (sample_size, dimension, noise)
    elif isinstance(sample_size, np.ndarray):
        independent_var = 'sample size'
        var_array = sample_size
        var_to_tuple = lambda sample_size: (sample_size, dimension, noise)
    elif isinstance(noise, np.ndarray):
        independent_var = 'noise'
        var_array = noise
        var_to_tuple = lambda noise: (sample_size, dimension, noise)
    else:
        sys.exit('Analyze called with incorrect syntax.')

    for var in var_array:
        devs_x = []
        devs_y = []
        for i in xrange(repeat):
            test_point = simulation_func(1,
                var if independent_var == 'dimension' else dimension,
                0)[0][0]
            training_x, training_y = simulation_func(*var_to_tuple(var))
            if population:
                x_opt = run_population(training_x, training_y)
            else:
                x_opt = run(training_x, training_y, test_point,
                            intervention_dim=correct_dims.size)
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
    xlim = (0, var_array[-1] * 1.03)
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
    # Define constants
    sample_size_array = np.array([10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500])
    dimensions_array = np.arange(2, 16)
    noise_array = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9])
    dimension = 10
    sample_size = 100
    noise = 0.2

    if len(sys.argv) <= 2:
        sys.exit('Usage: ./gp_gpy.py [--population] runs simulation1 simulation2 ...')
    # Parse options
    flags = set()
    args = []
    for arg in sys.argv:
        if arg[:2] == "--":
            flags.add(arg[2:])
        else:
            args.append(arg)
    repeat = int(args[1])
    pop = ""
    if "population" in flags:
        pop = ", population=True"

    # Execute trials
    for trial in args[2:]:
        exec "analyze(sample_size_array, dimension, noise, repeat, None, {trial}{pop})".format(trial=trial, pop=pop)
        #exec "analyze(sample_size, dimensions_array, noise, repeat, 'plots/{trial}_d.png', {trial}{pop})".format(trial=trial, pop=pop)
        #exec "analyze(sample_size, dimension, noise_array, repeat, 'plots/{trial}_n.png', {trial}{pop})".format(trial=trial, pop=pop)

if __name__ == "__main__":
    main()

