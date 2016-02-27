from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs
from gp_lib import mvn_logpdf
from simulation_data import parabola
from operator import add
import GPy
import GPyOpt

def find_sparse_intervention(objective, test_point, intervention_dim=1):
    def restricted_objective(dims):
        def fun(x):
            guess = np.copy(test_point)
            guess[dims] = x
            return -objective(guess)
        return fun
    def regularized_objective(l):
        return lambda x: -objective(x) + l * np.linalg.norm(x - test_point, ord=1)

    # Identify dimensions to be optimized
    reg_constants = [0]#np.power(10.0, np.arange(-7, 7)).tolist()
    log_search = True
    last_dim_diff = None
    x_opt = None
    while len(reg_constants) > 0:
        l = reg_constants.pop()
        print("Trying lambda=:", l)
        x_opt = fmin_bfgs(regularized_objective(l), test_point, disp=False)
        print(x_opt)
        if np.count_nonzero(x_opt != test_point) > intervention_dim:
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

def gpyopt_intervention(objective, test_point):
    D = len(test_point)
    def objective_wrapper(x):
        x = x.reshape((x.size/D, D))
        n = x.shape[0]
        return np.apply_along_axis(lambda x: -objective(x), 1, x).reshape(n, 1)
    opt = GPyOpt.methods.BayesianOptimization(f=objective_wrapper, bounds=[(-2,2) for i in xrange(test_point.size)])
    opt.run_optimization(20)
    #opt.plot_acquisition()
    #opt.plot_convergence()
    return opt.x_opt

def validate_gpy():
    D = 5
    n_range = range(1010, 9, -200)
    n_heldout = 1000
    fixed_x, fixed_y, z = parabola(n_heldout, D)

    for i in xrange(len(n_range)):
        n = n_range[i]
        print("Sample size", n)
        training_x, training_y, z = parabola(n, D)
        y = training_y.flatten()
        kernel = GPy.kern.RBF(D, ARD=True)
        model = GPy.models.GPRegression(training_x, training_y, kernel)
        model.optimize()
        y_pred, sigma2_pred = model.predict(fixed_x)

        #model.plot(fixed_inputs=[(i,0) for i in xrange(D-1)])
        #plt.show()
        print("MSE: ", np.mean( (fixed_y.flatten()-y_pred.flatten()) ** 2))
        print("Likelihood function:", mvn_logpdf(fixed_y.flatten(), y_pred.flatten(), np.diag(sigma2_pred.flatten())))
        print(kernel.lengthscale)

def run(training_x, training_y, test_point, plot_points=False, intervention_dim=1):
    D = training_x.shape[1]
    print("Running with sample size {ss} in a {d}-D space".format(ss=training_x.shape[0], d=D))

    x, y = training_x, training_y
    y = y.flatten()

    kernel = GPy.kern.RBF(D, ARD=True)
    model = GPy.models.GPRegression(training_x, training_y, kernel)

    print("Optimizing gp parameters.")
    model.optimize()

    print("Finding optimal intervention.")
    # 5th percentile of gp
    def objective(x_star):
        x = x_star.reshape((1, x_star.size))
        y_mu, y_cov = model.predict(x)
        ret = y_mu - 1.645 * np.sqrt(y_cov)
        return ret[0][0]

    x_opt = find_sparse_intervention(objective, test_point, intervention_dim=D)
    #x_opt = gpyopt_intervention(objective, test_point)
    print("Optimized value of x:", x_opt)

    if plot_points:
        model.plot(fixed_inputs=[(i,0) for i in xrange(D-1)])
        plt.show()

    return x_opt

def analyze_sample_size(sample_sizes, dim, simulation, simulation_eval, correct_dims, repeat):
    average_deviations = []
    std_devs = []
    correct_dim_found_count = 0
    for sample_size in sample_sizes:
        deviations = []
        for i in xrange(repeat):
            test_point = simulation(1, dim)[0][0]
            training_x, training_y, plot_points = simulation(sample_size, dim)
            x_opt = run(training_x, training_y, test_point, intervention_dim=dim)
            if np.all((x_opt != test_point)[correct_dims]):
                correct_dim_found_count += 1
            deviations.append(simulation_eval(x_opt))
        average_deviations.append(sum(deviations) / repeat)
        std_devs.append(np.std(deviations))
    print("Correct dimension to intervene on was identified in {} out of {} runs.".format(correct_dim_found_count, len(sample_sizes) * repeat))
    plt.figure()
    plt.title('intervention distance vs sample size')
    plt.xlabel('sample size')
    plt.ylabel('intervention distance')
    plt.plot(sample_sizes, average_deviations, ls='-', marker='+')
    plt.errorbar(sample_sizes, average_deviations, yerr=np.array(std_devs)*2)
    plt.show()

def analyze_dimensions(sample_size, dims, simulation, simulation_eval, correct_dims, repeat):
    average_deviations = []
    std_devs = []
    correct_dim_found_count = 0
    for dim in dims:
        deviations = []
        for i in xrange(repeat):
            test_point = simulation(1, dim)[0][0]
            training_x, training_y, plot_points = simulation(sample_size, dim)
            x_opt = run(training_x, training_y, test_point, intervention_dim=dim)
            if np.all((x_opt != test_point)[correct_dims]):
                correct_dim_found_count += 1
            deviations.append(simulation_eval(x_opt))
        average_deviations.append(sum(deviations) / repeat)
        std_devs.append(np.std(deviations))
    print("Correct dimension to intervene on was identified in {} out of {} runs.".format(correct_dim_found_count, len(dims) * repeat))
    #y = [min(abs(x_opt[-1] - 2.5), abs(x_opt[-1] - 0.5)) for x_opt in x_opts]    # For sine
    plt.figure()
    plt.title('intervention distance vs dimensions')
    plt.xlabel('dimensions')
    plt.ylabel('intervention distance')
    plt.plot(dims, average_deviations, ls='-', marker='+')
    plt.errorbar(dims, average_deviations, yerr=np.array(std_devs)*2)
    plt.show()

analyze_dimensions(80, np.arange(2, 21), parabola, lambda x: np.abs(x[-1]), np.array([-1]), 40)
#validate_gpy()
