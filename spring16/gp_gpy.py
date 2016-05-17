#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import numpy.ma as ma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from simulation import *
from plotting import *
from validation import check_gradient
from my_predictive_gradients import my_predictive_gradient
from PopulationIntervention import gradDesSoftThresholdBacktrack, sparsePopulationShift, sparsePopulationUniform
from itertools import compress
import GPy
import sys

def find_sparse_intervention(objective_and_grad, test_point, initial_guess,
                             intervention_dim=None, jac=True, constraint_bounds=None):
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
            can differ from test_point. If None, optimization is unconstrained.

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

    negate_tuple = lambda tup: tuple(-x for x in tup)
    negative_objective = lambda x: negate_tuple(objective_and_grad(x))

    def custom_minimize(objective, initial, constraint_bounds, test_point):
        if constraint_bounds is not None:
            return minimize(objective,
                         initial,
                         method='SLSQP',
                         jac=True,
                         bounds=constraint_bounds,
                         options={'disp':False}).x
        else:
            return minimize(objective,
                         initial,
                         method='BFGS',
                         jac=True,
                         options={'disp':False}).x

    if intervention_dim is None:
        return custom_minimize(negative_objective, initial_guess, constraint_bounds, test_point)

    l = 100.0
    last_dim_diff = None
    x_opt = None
    # Initialize upper bound
    while True:
        print("Trying l=", l)
        x_opt = gradDesSoftThresholdBacktrack(negative_objective, None, test_point, initial_guess, l, constraint_bounds)[0]
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
        x_opt = gradDesSoftThresholdBacktrack(negative_objective, None, test_point, initial_guess, l, constraint_bounds)[0]
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
        cb = None
        if constraint_bounds is not None:
            cb = list(compress(constraint_bounds,last_dim_diff))
        restricted_opt = custom_minimize(restricted_objective(last_dim_diff),
                                  initial_guess[last_dim_diff],
                                  cb,
                                  test_point[last_dim_diff])
        ret = np.copy(test_point)
        ret[last_dim_diff] = restricted_opt
        return ret

"""
def gd_soft_threshold(objective_and_grad, center, guess, l, constrained):
    prev = float('inf')
    diff = guess - center
    iteration = 0
    prev_grad = 0
    eta = 1.
    beta = 0.5
    while True:
        o, grad = objective_and_grad(diff + center)
        o += l * np.linalg.norm(diff, ord=1)
        if prev - o < 1e-9:
            return diff + center
        elif prev - o < (eta/2) * np.linalg.norm(prev_grad)**2:
            eta = eta * beta
        prev = o
        prev_grad = grad
        update = grad * eta
        diff = diff - update
        if constrained:
            diff = np.sign(diff) * np.minimum(np.maximum(np.abs(diff) - l * eta, 0), 1)
        else:
            diff = np.sign(diff) * np.maximum(np.abs(diff) - l * eta, 0)
        iteration += 1
"""

def train_gp(training_x, training_y):
    x, y = training_x, training_y
    y = y.flatten()
    D = training_x.shape[1]
    print("=" * 60)
    print("Training gp with sample size {ss} in a {d}-D space".format(
        ss=training_x.shape[0], d=D))

    kernel = GPy.kern.RBF(D, ARD=True)
    model = GPy.models.GPRegression(training_x, training_y, kernel)

    model.optimize()
    return model

def run(model, test_point, correct_dims=None, plot=False,
        constrained=False, smoothing=False, sparsity=None, restarts=False, mean_acq=False):
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
        intervention_dim (int): Max number of dimensions in which the
            intervention can differ from test_point. If None, optimization is
            unconstrained.

    Returns:
        numpy.ndarray: 1D array representing the optimal intervention.
    """
    FIVE_PERCENTILE_ZSCORE = -1.645
    D = test_point.size
    if plot:
        correct_dims = set(dim_index if dim_index >= 0 else D + dim_index\
            for dim_index in correct_dims)
        gpy_plot_fixed_inputs = [(i, 0) for i in xrange(D) if i not in correct_dims]

    def get_objective_and_grad(test_point):
        if mean_acq:
            def objective_and_grad(x_star):
                x = x_star.reshape((1, x_star.size))
                y_mu, y_var = model.predict(x, include_likelihood=False)
                y_mu = y_mu[0][0]
                dmu_dX, dv_dX = model.predictive_gradients(x)
                dmu_dX = dmu_dX.reshape((dmu_dX.size,))
                return y_mu, dmu_dX
            return objective_and_grad
        def objective_and_grad(x_star):
            x = np.vstack((x_star, test_point))
            y_mu, y_var = model.predict(x, full_cov=True, include_likelihood=False)
            y_mu = y_mu[0][0] - y_mu[1][0]
            y_var = y_var[0][0] + y_var[1][1] - 2.0 * y_var[1][0]
            if (y_var < 0) or (np.linalg.norm(test_point - x_star) < 1e-15):
                y_var = 0
            yvar_root = np.sqrt(y_var)
            ret = y_mu + FIVE_PERCENTILE_ZSCORE * yvar_root
            dmu_dX, dv_dX = my_predictive_gradient(x_star.reshape(1, x_star.size), model, test_point)
            dmu_dX = dmu_dX.reshape((dmu_dX.size,))
            dv_dX = dv_dX[0]
            if yvar_root < 1e-9:
                yvar_root = 1
            grad = dmu_dX + FIVE_PERCENTILE_ZSCORE * (0.5 / yvar_root) * dv_dX
            return ret, grad
        return objective_and_grad

    def acq(x, y):
        fmu, fv = model._raw_predict(np.array([[x, y]]))
        ret, = model.likelihood.predictive_quantiles(fmu, fv, (5,))
        return ret[0][0]

    print("Searching for optimal intervention using trained gp.")
    print("Test point:", test_point)
    print()

    # Optimization restarts routine for testing
    if restarts:
        print("Optimizing with optimization restarts and no sparsity")
        best_o = -float('inf')
        best_point = None
        for i in xrange(10):
            objective_and_grad = get_objective_and_grad(test_point)
            opt = find_sparse_intervention(objective_and_grad, test_point,
                    np.array([np.random.uniform(a[0], a[1]) for a in zip(test_point-1, test_point+1)]),
                    constraint_bounds=zip(test_point-1,test_point+1) if constrained else None)
            o, g = objective_and_grad(opt)
            if o > best_o:
                best_o = o
                best_point = opt
        print("We found that the best point is", best_point)
        return best_point

    # Smoothing
    kernel = model.kern
    orig_lengthscale = kernel.lengthscale.copy()
    orig_variance = kernel.variance.copy()
    orig_noise_var = model.likelihood.variance.copy()
    current_guess = test_point
    model.kern.lengthscale.fix()
    smoothing_values = (4, 2, 1) if smoothing else (1,)
    for i in smoothing_values:
        if smoothing:
            model.kern.lengthscale = i * orig_lengthscale
            model.optimize()

        new_bounds = None
        if i > 1:
            if constrained:
                low = np.maximum(test_point-1, np.amin(model.X, axis=0))
                high = np.minimum(test_point+1, np.amax(model.X, axis=0))
                new_bounds = zip(low, high)
            else:
                new_bounds = zip(np.amin(model.X, axis=0), np.amax(model.X, axis=0))
        elif constrained:
            new_bounds = zip(test_point-1, test_point+1)

        objective_and_grad = get_objective_and_grad(test_point)
        opt = find_sparse_intervention(
            objective_and_grad, test_point, current_guess,
            intervention_dim=sparsity, constraint_bounds=new_bounds)
        print("Test point will be reinitialized to", opt)

        current_guess = opt
        #print("Acquisition and gradient:", objective_and_grad(current_guess))

        if plot:
            model.plot(fixed_inputs=gpy_plot_fixed_inputs)
            plt.show()

        print()
        model.kern.variance = orig_variance
        model.likelihood.variance = orig_noise_var

    return current_guess

def analyze(sample_size, dimension, noise, repeat, file_prefix, simulation,
            kwargs_set, constrained=False, wishart=False):
    simulation_func, simulation_eval, true_opt, correct_dims = simulation

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
    elif wishart:
        independent_var = 'dof'
        var_array = 11 + np.arange(7) * 3
    else:
        sys.exit('Analyze called with incorrect syntax.')

    test_points = None
    if wishart:
        test_points = 2 * np.random.rand(repeat, 10) - 1
    else:
        test_points = simulation_func(repeat,
            dimension[-1] if independent_var == 'dimension' else dimension,
            0)[0]

    true_opts = np.apply_along_axis(true_opt, 1, test_points)
    initial_vals = np.array([simulation_eval(test_point) for test_point in test_points])

    np.save(file_prefix + "_true_opts", true_opts)
    np.save(file_prefix + "_initial_vals", initial_vals)

    models = {}

    for kwargs in kwargs_set:
        print("Running with", kwargs)
        avg_y_gain = []
        percentile_5_y_gain = []
        std_devs_y = []
        correct_dim_found_count = 0
        f_values = np.zeros((len(var_array), repeat))

        for j in xrange(len(var_array)):
            var = var_array[j]
            y_gain = []
            for i in xrange(repeat):
                training_x = None
                training_y = None
                if wishart:
                    training_x, training_y = simulation_func(sample_size, dimension, noise, var)
                else:
                    training_x, training_y = simulation_func(*var_to_tuple(var))
                model_key = str(var) + '_' + str(i)
                model = None
                if model_key in models:
                    model = models[model_key]
                else:
                    model = train_gp(training_x, training_y)
                    models[model_key] = model
                test_point = test_points[i]
                if independent_var == "dimension":
                    if 0 in correct_dims:
                        test_point = test_point[:var]
                    else:
                        test_point = test_point[-var:]
                initial_val = initial_vals[i]
                x_opt = run(model, test_point, constrained=constrained,
                            correct_dims=correct_dims, **kwargs)
                if np.all((x_opt != test_point)[correct_dims]):
                    correct_dim_found_count += 1
                final_val = simulation_eval(x_opt)
                f_values[j][i] = final_val
                y_gain.append(final_val - initial_val)
            avg_y_gain.append(sum(y_gain) / repeat)
            percentile_5_y_gain.append(np.percentile(y_gain, 5))
            std_devs_y.append(np.std(y_gain))

        filename = file_prefix + "".join(['_' + key for key in kwargs])
        np.save(filename, f_values)

        print("Correct intervention dimension was identified in {} out of {} runs."\
              .format(correct_dim_found_count, len(var_array) * repeat))
        xlim = (var_array[0] - 0.03 * var_array[-1], var_array[-1] * 1.03)
        plt.figure()
        plt.title('Objective gain vs {}'.format(independent_var))
        plt.xlabel(independent_var)
        plt.ylabel('objective gain')
        plt.xlim(xlim)
        plt.plot(var_array, avg_y_gain, ls='-', color='k', label='mean gain')
        plt.plot(var_array, percentile_5_y_gain, ls='-', color='b', label='5th percentile gain')
        iv = initial_vals
        to = true_opts
        horiz_lines = [np.average(to - iv), np.percentile(to - iv, 5)]
        plt.hlines(horiz_lines, xlim[0], xlim[1], colors=['k', 'b'], linestyles='dashed')
        #plt.legend()

        if file_prefix:
            plt.savefig(filename)
        else:
            plt.show()
        plt.close()

def analyze_population(sample_size, dimension, noise, repeat, file_prefix, simulation,
            constrained=False):
    simulation_func, simulation_eval, true_opt, correct_dims = simulation

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
    else:
        sys.exit('Analyze called with incorrect syntax.')

    models = {}
    avg_y_gain = []
    percentile_5_y_gain = []
    std_devs_y = []
    correct_dim_found_count = 0
    f_values = np.zeros((len(var_array), repeat))

    for j in xrange(len(var_array)):
        var = var_array[j]
        y_gain = []
        for i in xrange(repeat):
            training_x = None
            training_y = None
            training_x, training_y = simulation_func(*var_to_tuple(var))
            model_key = str(var) + '_' + str(i)
            model = None
            if model_key in models:
                model = models[model_key]
            else:
                model = train_gp(training_x, training_y)
                models[model_key] = model
            D = var if independent_var == "dimension" else dimension
            test_point = np.zeros(D)
            if independent_var == "dimension":
                test_point = test_point[:var]
            initial_val = simulation_eval(test_point) if constrained else 1.0/3
            x_opt = None
            if constrained:
                x_opt = sparsePopulationShift(model.X, model, cardinality=correct_dims.size,
                                              constraint_bounds=[(-2,2) for k in xrange(D)], smoothing_levels=(4,2,1))[0]
            else:
                x_opt = sparsePopulationUniform(model.X, model, cardinality=correct_dims.size,
                                              constraint_bounds=[(-2,2) for k in xrange(D)], smoothing_levels=(4,2,1))[0]
                if x_opt[-1] is ma.masked:
                    x_opt[-1] = 0.57735027
                if x_opt[-2] is ma.masked:
                    x_opt[-2] = 0.57735027
                x_opt = ma.filled(x_opt, fill_value=0)
            if np.all((x_opt != test_point)[correct_dims]):
                correct_dim_found_count += 1
            final_val = simulation_eval(x_opt)
            f_values[j][i] = final_val
            y_gain.append(final_val - initial_val)
        avg_y_gain.append(sum(y_gain) / repeat)
        percentile_5_y_gain.append(np.percentile(y_gain, 5))
        std_devs_y.append(np.std(y_gain))

    filename = file_prefix + '_population'
    np.save(filename, f_values)

    print("Correct intervention dimension was identified in {} out of {} runs."\
          .format(correct_dim_found_count, len(var_array) * repeat))
    xlim = (var_array[0] - 0.03 * var_array[-1], var_array[-1] * 1.03)
    plt.figure()
    plt.title('Objective gain vs {}'.format(independent_var))
    plt.xlabel(independent_var)
    plt.ylabel('objective gain')
    plt.xlim(xlim)
    plt.plot(var_array, avg_y_gain, ls='-', color='k', label='mean gain')
    plt.plot(var_array, percentile_5_y_gain, ls='-', color='b', label='5th percentile gain')
    test_point = np.zeros(var_array[-1] if independent_var == "dimension" else dimension)
    iv = np.ones(repeat) * (simulation_eval(test_point) if constrained else 1.0/3)
    to = np.ones(repeat) * true_opt(test_point)
    horiz_lines = [np.average(to - iv), np.percentile(to - iv, 5)]
    plt.hlines(horiz_lines, xlim[0], xlim[1], colors=['k', 'b'], linestyles='dashed')
    #plt.legend()

    if file_prefix:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()

def main():
    # Define constants
    sample_size_array = np.array([20, 30, 40, 50, 75, 100, 150, 200])
    dimensions_array = np.arange(2, 16)
    dimension = 10
    sample_size = 100
    noise = 0.2

    if len(sys.argv) > 1 and sys.argv[1][:2] == "--":
        if sys.argv[1][2:] != "population":
            sys.exit("Flag not recognized")
        repeat = 8
#        analyze_population(sample_size_array, dimension, noise, repeat, 'plots/paraboloid_ss', paraboloid)
#        analyze_population(sample_size_array, dimension, noise, repeat, 'plots/line_ss', line, constrained=True)
        analyze_population(sample_size_array, dimension, noise, repeat, 'plots/plane_ss', plane, constrained=True)
        return

    # Execute trials
    repeat = 100
    analyze(sample_size, dimension, noise, repeat, 'plots/wishart_dof', wishart_paraboloid, [dict(smoothing=True), dict(smoothing=True, sparsity=2), dict(restarts=True), dict(mean_acq=True), dict()], wishart=True)
    analyze(sample_size_array, dimension, noise, repeat, 'plots/hyperbolic_ss', hyperbolic, [dict(smoothing=True, sparsity=1), dict(smoothing=True, mean_acq=True, sparsity=1)], constrained=True)
    analyze(sample_size_array, dimension, noise, repeat, 'plots/paraboloid_ss', paraboloid, [dict(smoothing=True), dict(smoothing=True, sparsity=2), dict(restarts=True), dict(mean_acq=True), dict()])
    analyze(sample_size_array, dimension, noise, repeat, 'plots/line_ss', line, [dict(smoothing=True), dict(smoothing=True, sparsity=1), dict(restarts=True), dict(mean_acq=True), dict()], constrained=True)
    analyze(sample_size_array, dimension, noise, repeat, 'plots/plane_ss', plane, [dict(smoothing=True), dict(smoothing=True, sparsity=2), dict(restarts=True), dict(mean_acq=True), dict()], constrained=True)

if __name__ == "__main__":
    main()

