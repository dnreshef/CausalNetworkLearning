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
from my_predictive_gradients import my_predictive_gradient

# TO run: use: ./test_my_gradient.py 1 parabola

def find_sparse_intervention(objective_and_grad, test_point, initial_guess,
                             intervention_dim=None, jac=True, constrained=False):
    """
    Finds an intervention constrained to differ from the test point in at most
    intervention_dim dimensions.

    Args:
        objective_and_grad (function): Accepts 1D array of length
            test_point.size, outputs objective, gradient (float, numpy.ndarray)
            The objective function is maximized.
        test_point (numpy.ndarray): point to find intervention on
        initial_guess (numpy.ndarray): 1D array representing the initial guess
        intervention_dim (int): Max number of dimensions in which intervention
            can differ from test_point. If none, optimization is unconstrained.

    Returns:
        numpy.ndarray: 1D array of length test_point.size, the optimal intervention
    """
    negate_tuple = lambda tup: tuple(-x for x in tup)
    def restricted_objective(dims):
        if jac:
            def fun(x):
                guess = np.copy(test_point)
                guess[dims] = x
                obj, grad = objective_and_grad(guess)
                return -obj, -grad[dims]
            return fun
        else:
            def fun(x):
                guess = np.copy(test_point)
                guess[dims] = x
                return -objective_and_grad(guess)
            return fun

    negative_objective = None
    if jac:
        negative_objective = lambda x: negate_tuple(objective_and_grad(x))
    else:
        negative_objective = lambda x: -objective_and_grad(x)

    if intervention_dim is None:
        #x_opt = minimize(negative_objective,
        #                 initial_guess,
        #                 method='BFGS',
        #                 jac=jac,
        #                 bounds=zip(test_point-2,test_point+2) if constrained else None,
        #                 options={'disp':False}).x
        l = 0.0
        x_opt = gd_soft_threshold(negative_objective, test_point, initial_guess, l)
        return x_opt
    #reg_constants = np.power(10.0, np.arange(-7, 7)).tolist()
    l = 100.0

    last_dim_diff = None
    x_opt = None
    # Initialize upper bound
    while True:
        print("Trying l=", l)
        x_opt = gd_soft_threshold(negative_objective, test_point, initial_guess, l)
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
        x_opt = gd_soft_threshold(negative_objective, test_point, initial_guess, l)
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

def gd_soft_threshold(objective_and_grad, center, guess, l):
    prev = float('inf')
    diff = guess - center
    iteration = 0
    prev_grad = 0
    eta = 0.2
    beta = 0.5
    while True:
        o, grad = objective_and_grad(diff + center)
        o += l * np.linalg.norm(diff, ord=1)
        if prev - o < 1e-9 and iteration >= 1e3:
            return diff + center
        elif prev - o < (eta/2) * np.linalg.norm(prev_grad)**2:
            eta = eta * beta
        prev = o
        prev_grad = grad
        update = grad * eta
        diff = diff - update
        diff = np.sign(diff) * np.maximum(np.abs(diff) - l * eta, 0)
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

    return find_sparse_intervention(population_objective_and_grad, np.zeros(D), np.zeros(D))

def run(training_x, training_y, test_point, plot_dims=None, intervention_dim=None, constrained=True):
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

    def get_objective_and_grad(test_point):
        def objective_and_grad(x_star):
            x = np.vstack((x_star, test_point))
            y_mu, y_var = model.predict(x, full_cov=True, include_likelihood=False)
            y_mu = y_mu[0][0] - y_mu[1][0]
            y_var = y_var[0][0] + y_var[1][1] - 2.0 * y_var[1][0]
            if (y_var < 0) or (np.linalg.norm(test_point - x_star) < 1e-15):
                y_var = 0
            yvar_root = np.sqrt(y_var) 
            ret = y_mu + FIVE_PERCENTILE_ZSCORE * yvar_root
            print('Objective: ',ret, ' mu-diff:', y_mu, ' var:', y_var)
            # return ret
            dmu_dX, dv_dX = my_predictive_gradient(x_star.reshape(1, x_star.size), model, test_point)
            dmu_dX = dmu_dX.reshape((dmu_dX.size,))
            dv_dX = dv_dX[0]
            if yvar_root < 1e-9:
                yvar_root = 19
            grad = dmu_dX + FIVE_PERCENTILE_ZSCORE * (0.5 / yvar_root) * dv_dX
            #if np.linalg.norm(x_star - test_point) < 1e-3:
            #print(np.linalg.norm(x_star - test_point))
            #print(y_mu)
            #print(y_var)
            return ret, grad
        return objective_and_grad

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

    # Optimization restarts routine for testing
    """
    print("Optimizing with optimization restarts and no sparsity")
    best_o = -float('inf')
    best_point = None
    for i in xrange(10):
        opt = find_sparse_intervention(objective_and_grad, test_point, x[np.random.randint(0, len(x))])
        o, g = objective_and_grad(opt)
        if o > best_o:
            best_o = o
            best_point = opt
    return best_point
    """

    # Smoothing
    orig_lengthscale = kernel.lengthscale.copy()
    orig_variance = kernel.variance.copy()
    current_guess = test_point
    model.kern.lengthscale.fix()
    for i in (1,):
        #model.kern.lengthscale = i * orig_lengthscale
        # Recompute variance and noise scale
        #model.optimize()

        objective_and_grad = get_objective_and_grad(test_point)
        opt = find_sparse_intervention(
            objective_and_grad, test_point, current_guess, intervention_dim, jac=True)
        print("Test point will be reinitialized to", opt)

        current_guess = opt
        #print("Acquisition and gradient:", objective_and_grad(current_guess))

        if plot_dims is not None:
            model.plot(fixed_inputs=gpy_plot_fixed_inputs)
            plt.show()

        print()
        model.kern.variance = orig_variance

    print()
    return current_guess

def analyze(sample_size, dimension, noise, repeat, filename, simulation,
            population=False):
    simulation_func, simulation_eval, correct_dims = simulation

    avg_y_gain = []
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

    test_points = simulation_func(repeat,
        var if independent_var == 'dimension' else dimension,
        0)[0]
    initial_errs = [simulation_eval(test_point) for test_point in test_points]

    for var in var_array:
        y_gain = []
        for i in xrange(repeat):
            test_point = test_points[i]
            initial_err = initial_errs[i]
            training_x, training_y = simulation_func(*var_to_tuple(var))
            if population:
                x_opt = run_population(training_x, training_y)
            else:
                x_opt = run(training_x, training_y, test_point)
#                            intervention_dim=correct_dims.size)
            if np.all((x_opt != test_point)[correct_dims]):
                correct_dim_found_count += 1
            final_err = simulation_eval(x_opt)
            y_gain.append(initial_err - final_err)
        avg_y_gain.append(sum(y_gain) / repeat)
        std_devs_y.append(np.std(y_gain))

    print("Correct intervention dimension was identified in {} out of {} runs."\
          .format(correct_dim_found_count, len(var_array) * repeat))
    xlim = (0, var_array[-1] * 1.03)
    plt.figure(num=None, figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')

    plt.title('objective difference vs {}'.format(independent_var))
    plt.xlabel(independent_var)
    plt.ylabel('objective difference')
    plt.xlim(xlim)
    plt.plot(var_array, avg_y_gain, ls='-', marker='o')
    plt.hlines(sum(initial_errs) / repeat, xlim[0], xlim[1], linestyles='dashed')
    #plt.errorbar(var_array, avg_y_gain, yerr=np.array(std_devs_y))

    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def main():
    # Define constants
    sample_size_array = np.array([50])
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
#        exec "analyze(sample_size, dimensions_array, noise, repeat, 'plots/{trial}_d.png', {trial}{pop})".format(trial=trial, pop=pop)
#        exec "analyze(sample_size, dimension, noise_array, repeat, 'plots/{trial}_n.png', {trial}{pop})".format(trial=trial, pop=pop)

if __name__ == "__main__":
    main()

