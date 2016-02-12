from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs
from scipy.optimize import minimize
from gp_lib import make_gp_funs, rbf_covariance, mvn_logpdf
from simulation_data import parabola, sine
from operator import add

def find_sparse_intervention(objective, test_point, dim=1):
    def restricted_objective(dims):
        def fun(x):
            guess = np.copy(test_point)
            guess[dims] = x
            return -objective(guess)
        return fun
    def regularized_objective(l):
        return lambda x: -objective(x) + l * np.linalg.norm(l - test_point, ord=1)

    # Identify dimensions to be optimized
    reg_constants = np.power(10.0, np.arange(-7, 7)).tolist()
    log_search = True
    last_dim_diff = None
    x_opt = None
    while len(reg_constants) > 0:
        l = reg_constants.pop()
        print("Trying lambda=:", l)
        x_opt = fmin_bfgs(regularized_objective(l), test_point, disp=False)
        print(x_opt)
        if np.count_nonzero(x_opt != test_point) > dim:
            assert last_dim_diff is not None
            if log_search:
                reg_constants = (np.arange(2, 10) * l).tolist()
                log_search = False
            else: 
                break
        else:
            last_dim_diff = x_opt != test_point

    # If dimensions found, now optimize
    if np.count_nonzero(last_dim_diff) == 0:
        print("I couldn't optimize the test point at all.")
        return test_point
    else: 
        print("Now optimizing these dimensions:",last_dim_diff)
        restricted_opt = fmin_bfgs(restricted_objective(last_dim_diff), test_point[last_dim_diff])
        ret = np.copy(test_point)
        ret[last_dim_diff] = restricted_opt
        return ret

def find_gp_parameters(num_params, nll):
    init_params = np.zeros(num_params)
    optimization_bounds = [(None, None)]+[(-15, 5) for i in range(num_params - 1)]
    return minimize(nll, init_params, method='L-BFGS-B', bounds=optimization_bounds)['x']

def validate_gp():
    D = 5
    fixed_x, fixed_y, z = parabola(1000, D)
    nlls = []
    for i in xrange(50, 301, 50):
        training_x, training_y, z = parabola(i, D)
        y = training_y.flatten()
        num_params, predict, log_marginal_likelihood = make_gp_funs(rbf_covariance, num_cov_params=D + 1)
        opt_params = find_gp_parameters(num_params, lambda params: -log_marginal_likelihood(params, training_x, y))
        pred_mean, pred_cov = predict(opt_params, training_x, y, fixed_x) 
        nlls.append(mvn_logpdf(fixed_y.flatten(), pred_mean, np.diag(np.diag(pred_cov))))
    print("log likelihoods:", nlls)

def run(training_x, training_y, test_point, plot_points=None):
    D = training_x.shape[1]
    print("Running with sample size {ss} in a {d}-D space".format(ss=training_x.shape[0], d=D))
    print("Optimizing gp parameters.")
    # Build model and objective function.
    num_params, predict, log_marginal_likelihood = make_gp_funs(rbf_covariance, num_cov_params=D + 1)

    x, y = training_x, training_y
    y = y.flatten()

    opt_params = find_gp_parameters(num_params, lambda params: -log_marginal_likelihood(params, x, y))
    print("Optimized paramaters:",opt_params)

    print("Finding optimal intervention.")
    # 5th percentile of gp
    def objective(x_star):
        ymu, y_cov = predict(opt_params, x, y, x_star.reshape((1, x_star.size)))
        ret = ymu - 1.645 * np.sqrt(np.diag(y_cov))
        return ret[0]

    x_opt = find_sparse_intervention(objective, test_point)
    print("Optimized value of x:", x_opt)

    if plot_points is None:
        print("\n")
        return x_opt

    ymu, y_cov = predict(opt_params, x, y, plot_points)
    ys2 = np.diag(y_cov)
    q_95 = ymu + 1.645 * np.sqrt(ys2)
    q_5 = ymu - 1.645 * np.sqrt(ys2)

    plt.figure()
    ymu = np.reshape(ymu, (ymu.shape[0],))
    plt.title('gp regression {d}d'.format(d=D))
    plt.plot(plot_points[:,-1], ymu, ls='None', marker='+')
    plt.fill_between(plot_points[:,-1], q_95, q_5, facecolor=[0.7539, 0.89453125, 0.62890625, 1.0], linewidths=0)
    opt_mu, opt_cov = predict(opt_params, x, y, x_opt.reshape((x_opt.size, 1)))
    plt.plot(x_opt[-1], opt_mu[-1] - 1.645 * np.sqrt(np.diag(opt_cov)[-1]), marker='.')
    plt.show()
    return x_opt

def analyze_sample_size(sample_sizes, dim):
    repeat = 20
    test_point = np.arange(0, 1.61, 1.6 / (dim-1))
    x_opts = [0] * len(sample_sizes)
    correct_dim_found_count = 0
    for i in xrange(repeat):
        x_opts_i = []
        for sample_size in sample_sizes:
            training_x, training_y, plot_points = sine(sample_size, dim)
            x_opt_i = run(training_x, training_y, test_point)
            if (x_opt_i != test_point)[-1]:
                correct_dim_found_count += 1
            x_opts_i.append(x_opt_i)
        x_opts = map(add, x_opts, x_opts_i)
    print("Correct dimension to intervene on was identified in {} out of {} runs.".format(correct_dim_found_count, len(sample_sizes) * repeat))
    x_opts = [r / repeat for r in x_opts]
    #y = [x_opt[-1] for x_opt in np.abs(x_opts)]
    y = [min(abs(x_opt[-1] - 2.5), abs(x_opt[-1] - 0.5)) for x_opt in x_opts]
    plt.figure()
    plt.title('intervention distance vs sample size')
    plt.xlabel('sample size')
    plt.ylabel('intervention distance')
    plt.plot(sample_sizes, y, ls='-', marker='+')
    plt.show()

def analyze_dimensions(sample_size, dims):
    repeat = 15
    x_opts = [0] * len(dims)
    correct_dim_found_count = 0
    for i in xrange(repeat):
        x_opts_i = []
        for dim in dims:
            test_point = np.arange(0, 1.61, 1.6 / (dim-1))
            training_x, training_y, plot_points = sine(sample_size, dim)
            x_opt_i = run(training_x, training_y, test_point)
            if (x_opt_i != test_point)[-1]:
                correct_dim_found_count += 1
            x_opts_i.append(x_opt_i)
        x_opts = map(add, x_opts, x_opts_i)
    print("Correct dimension to intervene on was identified in {} out of {} runs.".format(correct_dim_found_count, len(dims) * repeat))
    x_opts = [r / repeat for r in x_opts]
    #y = [x_opt[-1] for x_opt in np.abs(x_opts)]
    y = [min(abs(x_opt[-1] - 2.5), abs(x_opt[-1] - 0.5)) for x_opt in x_opts]
    plt.figure()
    plt.title('intervention distance vs dimensions')
    plt.xlabel('dimensions')
    plt.ylabel('intervention distance')
    plt.plot(dims, y, ls='-', marker='+')
    plt.show()

#analyze_sample_size(np.arange(100, 601, 100), 10)
#analyze_dimensions(500, np.arange(2, 16, 1))
validate_gp()
