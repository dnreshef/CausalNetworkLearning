from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs
from scipy.optimize import minimize
from gp_lib import make_gp_funs, rbf_covariance, mvn_logpdf
from simulation_data import parabola, sine
from operator import add
from sklearn import gaussian_process

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

def find_gp_parameters(num_params, nll):
    init_params = np.zeros(num_params)
    init_params[1] = 0
    print("Init parameters: ", init_params)

    # When parameters are optimized, the MSE decreases for increasing sample size, but for some reason the log likelihood decreases (perhaps due to overfitting because the predicted variance is getting way too small).
    optimization_bounds = [(None, None)]+[(-15, 15) for i in range(num_params - 1)]
    #optimized_params = minimize(nll, init_params, method='L-BFGS-B', bounds=optimization_bounds)['x']
    #optimized_restricted_params = minimize(lambda params: nll(np.concatenate((params,init_params[3:]))), init_params[:3], method='L-BFGS-B', bounds=optimization_bounds[:3])['x']
    #optimized_params = np.concatenate((optimized_restricted_params, init_params[3:]))
    #print("Optimized parameters: ", optimized_params)
    #return optimized_params
    return init_params

def validate_gp():
    nrep = 1
    D = 1
    n_range = range(510, 9, -500)
    n_heldout = 1000
    avg_nlls = [0]*len(n_range)
    avg_mses = [0]*len(n_range)
    for rep in xrange(nrep):
        fixed_x, fixed_y, z = parabola(n_heldout, D)
        nlls = [0]*len(n_range)
        mses = [0]*len(n_range)
        for i in xrange(len(n_range)):
            n = n_range[i]
            training_x, training_y, z = parabola(n, D)
            y = training_y.flatten()
            num_params, predict, log_marginal_likelihood, avg_heldout_loglik = make_gp_funs(rbf_covariance, num_cov_params=D + 1)
            if(i==0):
                opt_params = find_gp_parameters(num_params, lambda params: -log_marginal_likelihood(params, training_x, y))
            pred_mean, pred_cov = predict(opt_params, training_x, y, fixed_x)
            nlls[i] = (mvn_logpdf(fixed_y.flatten(), pred_mean, np.diag(np.diag(pred_cov)))/n_heldout)
            #nlls[i] = avg_heldout_loglik(opt_params, training_x,y,fixed_x,fixed_y.flatten())
            print("Predictive Cov: ", np.mean((np.diag(pred_cov))))
            mses[i] = (np.mean( (fixed_y.flatten()-pred_mean) ** 2))

            avg_mses[i] = avg_mses[i] + mses[i]
            avg_nlls[i] = avg_nlls[i] + nlls[i]

            # Plot output
            #plt.plot(fixed_x[:,-1], fixed_y, ls='None', marker='+')
            #plt.plot(fixed_x[:,-1], pred_mean, ls='None', c='red', marker='*')
            #plt.show()

        # Single trial output
        #print("log likelihoods:", nlls)
        #print("Mean Squared Errors:", mses)

    # Multi trial output
    avg_mses = [x / nrep for x in avg_mses]
    avg_nlls = [x / nrep for x in avg_nlls]
    print("avg log likelihoods:", avg_nlls)
    print("avg Mean Squared Errors:", avg_mses)


def f(x):
    return np.power(x,2)

def validate_gp_scikitlearn():
    X = np.atleast_2d(np.linspace(-1, 1, 50)).T
    y = f(X).ravel() + 0.2 * np.random.standard_normal()
    x = np.atleast_2d(np.linspace(-1, 1, 1000)).T
    gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
    gp.fit(X, y)
    y_pred, sigma2_pred = gp.predict(x, eval_MSE=True)
    print("MSE: ", np.mean( (f(x).ravel()-y_pred) ** 2))
    print("Sigma: ", np.mean(sigma2_pred))

def validate_gp_scikitlearn2():
    D = 1
    n_range = range(510, 9, -500)
    n_heldout = 1000
    fixed_x, fixed_y, z = parabola(n_heldout, D)

    for i in xrange(len(n_range)):
        n = n_range[i]
        training_x, training_y, z = parabola(n, D)
        y = training_y.flatten()
        gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
        gp.fit(training_x, training_y)
        y_pred, sigma2_pred = gp.predict(fixed_x, eval_MSE=True)

        print("MSE: ", np.mean( (fixed_y.flatten()-y_pred) ** 2))
        print("Sigma: ", sigma2_pred)




def run(training_x, training_y, test_point, plot_points=None, intervention_dim=1):
    D = training_x.shape[1]
    print("Running with sample size {ss} in a {d}-D space".format(ss=training_x.shape[0], d=D))
    print("Optimizing gp parameters.")
    # Build model and objective function.
    num_params, predict, log_marginal_likelihood, avg_heldout_loglik = make_gp_funs(rbf_covariance, num_cov_params=D + 1)

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

    x_opt = find_sparse_intervention(objective, test_point, intervention_dim)
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
            x_opt_i = run(training_x, training_y, test_point, intervention_dim=dim)
            if (x_opt_i != test_point)[-1]:
                correct_dim_found_count += 1
            x_opts_i.append(x_opt_i)
            print("Test point: ", test_point, " X_optimal: ", x_opt_i)
        x_opts = map(add, x_opts, x_opts_i)
    print("Correct dimension to intervene on was identified in {} out of {} runs.".format(correct_dim_found_count, len(sample_sizes) * repeat))
    x_opts = [r / repeat for r in x_opts]
    y = [x_opt[-1] for x_opt in np.abs(x_opts)]                                     # For parabola
    #y = [min(abs(x_opt[-1] - 2.5), abs(x_opt[-1] - 0.5)) for x_opt in x_opts]      # For sine
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
            x_opt_i = run(training_x, training_y, test_point, intervention_dim=dim)
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

#analyze_sample_size(np.arange(100, 501, 100), 20)
#analyze_dimensions(500, np.arange(2, 16, 1))
#validate_gp()
validate_gp_scikitlearn()
