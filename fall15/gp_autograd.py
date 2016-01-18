import autograd.numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs
from autograd.numpy.linalg import solve
import autograd.scipy.stats.multivariate_normal as mvn
from scipy.optimize import minimize

def make_gp_funs(cov_func, num_cov_params):
    """Functions that perform Gaussian process regression.
       cov_func has signature (cov_params, x, x')"""

    def unpack_params(params):
        mean        = params[0]
        cov_params  = params[2:]
        noise_scale = np.exp(params[1]) + 0.001
        return mean, cov_params, noise_scale

    def predict(params, x, y, xstar):
        """Returns the predictive mean and covariance at locations xstar,
           of the latent function value f (without observation noise)."""
        mean, cov_params, noise_scale = unpack_params(params)
        cov_f_f = cov_func(cov_params, xstar, xstar)
        cov_y_f = cov_func(cov_params, x, xstar)
        cov_y_y = cov_func(cov_params, x, x) + noise_scale * np.eye(len(y))
        pred_mean = mean +   np.dot(solve(cov_y_y, cov_y_f).T, y - mean)
        pred_cov = cov_f_f - np.dot(solve(cov_y_y, cov_y_f).T, cov_y_f)
        return pred_mean, pred_cov

    def log_marginal_likelihood(params, x, y):
        mean, cov_params, noise_scale = unpack_params(params)
        cov_y_y = cov_func(cov_params, x, x) + noise_scale * np.eye(len(y))
        prior_mean = mean * np.ones(len(y))
        return mvn.logpdf(y, prior_mean, cov_y_y)

    return num_cov_params + 2, predict, log_marginal_likelihood

# Define an example covariance function.
def rbf_covariance(kernel_params, x, xp):
    output_scale = np.exp(kernel_params[0])
    lengthscales = np.exp(kernel_params[1:])
    diffs = np.expand_dims(x /lengthscales, 1)\
          - np.expand_dims(xp/lengthscales, 0)
    return output_scale * np.exp(-0.5 * np.sum(diffs**2, axis=2))

def generate_toy_data(sample_size=100, D=2):
    x = 2 * np.random.rand(sample_size, D) - 1 # D-dimensional, uniform in [-1, 1)
    f_x = lambda x: [1 - x[-1] ** 2 + 0.1 * np.random.standard_normal()]
    y = np.apply_along_axis(f_x, 1, x)
    z = 2 * np.random.rand(sample_size, D) - 1 # D-dimensional, uniform in [-1, 1)
    return x, y, z # train x, train y, test x

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
    print x.shape
    print top.shape
    print bottom.shape
    t = np.column_stack((x, top, bottom))
    return t[t[:,0].argsort()]

def run(sample_size, D, plot=False):
    # Build model and objective function.
    num_params, predict, log_marginal_likelihood = \
        make_gp_funs(rbf_covariance, num_cov_params=D + 1)

    x, y, z = generate_toy_data(sample_size=sample_size, D=D)
    y = y.flatten()
    nll = lambda params: -log_marginal_likelihood(params, x, y)

    lel = np.apply_along_axis(np.std, 0, x)
    lengthscale = edistance_at_percentile(x, 50) # output scale

    init_params = np.hstack((np.array([0, 0, lengthscale]), lel))
    print("Paramaters: {}".format(init_params))
    # Value and grad doesn't work
    cov_params = minimize(nll, init_params)['x']
    print("Optimized paramaters: {}".format(cov_params))

    def objective(x_star):
        ymu, ys2 = predict(cov_params, x, y, x_star.reshape((1, len(x_star))))
        ret = ymu - 1.645 * np.sqrt(ys2)
        return ret[0][0]

    x_opt = fmin_bfgs(lambda x: objective(x) * -1, np.arange(0, 0.2, 0.1))
    print "Optimized value of x:", x_opt

    if not plot:
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

x = np.arange(50, 500, 25)
x_opts = [run(sample_size, 2) for sample_size in x]
y = [np.absolute(x_opt[-1]) for x_opt in x_opts]
plt.figure()
plt.plot(x, y, ls='-', marker='+')
plt.show()
