import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs
from scipy.optimize import minimize
from gp_lib import make_gp_funs, rbf_covariance
from simulation_data import parabola

def edistance_at_percentile(X, q):
    distances = []
    for i in xrange(10000):
        a = X[np.random.choice(len(X))]
        b = X[np.random.choice(len(X))]
        distances.append(np.linalg.norm(a - b))
    distances.sort()
    for i in xrange(q*100, 10000):
        if distances[i] > 0:
            return distances[i]

def sort_for_plotting(x, top, bottom):
    t = np.column_stack((x, top, bottom))
    return t[t[:,0].argsort()]

def find_sparse_intervention(objective, init_guess, dim=1):
    def regularized_objective(l):
        return lambda x: -objective(x) + l * np.linalg.norm(l - init_guess, ord=1)
    reg_constants = np.power(10.0, np.arange(-7, 7)).tolist()
    log_search = True
    last_intervention = None
    x_opt = None
    while len(reg_constants) > 0:
        l = reg_constants.pop()
        print("Trying lambda={l}:".format(l=l))
        x_opt = fmin_bfgs(regularized_objective(l), init_guess)
        print x_opt
        if np.count_nonzero(x_opt != init_guess) > dim:
            assert last_intervention is not None
            if log_search:
                reg_constants = (np.arange(2, 10) * l).tolist()
                log_search = False
            else:
                return last_intervention
        else:
            last_intervention = x_opt
    return x_opt

def run(sample_size, D, plot=False):
    print("Running with sample size {ss} in a {d}-D space".format(ss=sample_size, d=D))
    # Build model and objective function.
    num_params, predict, log_marginal_likelihood = make_gp_funs(rbf_covariance, num_cov_params=D + 1)

    x, y, z = parabola(sample_size=sample_size, D=D)
    y = y.flatten()
    nll = lambda params: -log_marginal_likelihood(params, x, y)

    #lel = np.apply_along_axis(np.std, 0, x)
    #lengthscale = edistance_at_percentile(x, 50) # output scale

    init_params = np.zeros(num_params)
    #print("Initial paramaters: {}".format(init_params))
    optimization_bounds = [(None, None)]+[(-15, 5) for i in range(num_params - 1)]
    cov_params = minimize(nll, init_params, method='L-BFGS-B', bounds=optimization_bounds)['x']
    print("Optimized paramaters: {}".format(cov_params))

    print("Now finding optimal intervention:")
    # 5th percentile of gp
    def objective(x_star):
        ymu, y_cov = predict(cov_params, x, y, x_star.reshape((1, x_star.size)))
        ret = ymu - 1.645 * np.sqrt(np.diag(y_cov))
        return ret[0]

    init_guess = np.arange(0, D * 0.1 - 0.01, 0.1)
    x_opt = find_sparse_intervention(objective, init_guess)
    print "Optimized value of x:", x_opt

    if not plot:
        print
        return x_opt

    ymu, y_cov = predict(cov_params, x, y, z)
    ys2 = np.diag(y_cov)
    q_95 = ymu + 1.645 * np.sqrt(ys2)
    q_5 = ymu - 1.645 * np.sqrt(ys2)
    t1 = sort_for_plotting(z[:,-1], q_95, q_5)
    t2 = sort_for_plotting(z[:,0], q_95, q_5)

    plt.figure()
    ymu = np.reshape(ymu, (ymu.shape[0],))
    plt.plot(z[:,-1], ymu, ls='None', marker='+')
    plt.fill_between(t1[:,0], t1[:,1], t1[:,2], facecolor=[0.7539, 0.89453125, 0.62890625, 1.0], linewidths=0)
    plt.show()
    plt.figure()
    plt.plot(z[:,0], ymu, ls='None', marker='+')
    plt.fill_between(t2[:,0], t2[:,1], t2[:,2], facecolor=[0.7539, 0.89453125, 0.62890625, 1.0], linewidths=0)
    plt.show()
    return x_opt

#x = 100 * np.ones(1)
x = np.arange(100, 1001, 25)
x_opts = [run(sample_size, 9) for sample_size in x]
y = [x_opt[-1] for x_opt in np.abs(x_opts)]
plt.figure()
plt.plot(x, y, ls='-', marker='+')
#plt.plot(100, np.average(y), color='r', marker='x')
plt.show()

