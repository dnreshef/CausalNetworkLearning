import numpy as np
from numpy.linalg import solve
from scipy.stats import norm

def make_gp_funs(cov_func, num_cov_params):
    """Functions that perform Gaussian process regression.
       cov_func has signature (cov_params, x, x')"""

    def unpack_params(params):
        mean        = params[0]
        noise_scale = np.exp(params[1])
        cov_params  = params[2:]
        #if params[1] <= 100:
        #    noise_scale = np.exp(params[1]) + 0.001
        #else:
        #    noise_scale = np.exp(100) # hack for optimizer to prevent overflow
        return mean, cov_params, noise_scale

    def predict(params, x, y, xstar):
        """Returns the predictive mean and covariance at locations xstar,
           of the latent function value f (without observation noise)."""
        mean, cov_params, noise_scale = unpack_params(params)
        cov_f_f = cov_func(cov_params, xstar, xstar)
        cov_y_f = cov_func(cov_params, x, xstar)
        cov_y_y = cov_func(cov_params, x, x) + noise_scale * np.eye(len(y))
        pred_mean = mean +   np.dot(solve(cov_y_y, cov_y_f).T, y - mean)
        print("Pred mean: ", np.mean(pred_mean))
        print("cov_f_f: ", np.mean(np.diag(cov_f_f)))
        print("cov_y_f: ", np.mean(np.diag(cov_y_f)))
        print("cov_y_y: ", np.mean(np.diag(cov_y_y)))
        pred_cov = cov_f_f - np.dot(solve(cov_y_y, cov_y_f).T, cov_y_f)
        return pred_mean, pred_cov

    def log_marginal_likelihood(params, x, y):
        mean, cov_params, noise_scale = unpack_params(params)
        cov_y_y = cov_func(cov_params, x, x) + noise_scale * np.eye(len(y))
        cov_y_y = np.round(cov_y_y, 10)
        prior_mean = mean * np.ones(len(y))
        try:
            ret = mvn_logpdf(y, prior_mean, cov_y_y)
            return ret
        except:
            print ("Cov:",cov_y_y)
            print ("Eigvals:",np.linalg.eigvals(cov_y_y))
            print ("Max:",np.absolute(cov_y_y).max())
            print ("Sum:",cov_y_y.sum())
            print ("Y:",y)
            print ("Mean:",prior_mean)
            print ("Params:",params)
            raise

    def avg_heldout_loglik(params, x, y, heldout_x, heldout_y):
        hlk = 0
        for i in xrange(heldout_x.shape[0]):
            pred_mean, pred_cov = predict(params, x,y, np.matrix(heldout_x[i,:]))
            hlk += mvn_logpdf(np.array([heldout_y[i]]),pred_mean, pred_cov)
            #hlk += norm.logpdf(heldout_y[i], pred_mean[0], pred_cov[0,0]**0.5)
        hlk /= heldout_x.shape[0]
        return hlk

    return num_cov_params + 2, predict, log_marginal_likelihood, avg_heldout_loglik

# Define an example covariance function.
def rbf_covariance(kernel_params, x, xp):
    output_scale = np.exp(kernel_params[0])
    lengthscales = np.exp(kernel_params[1:])
    diffs = np.expand_dims(x /lengthscales, 1)\
          - np.expand_dims(xp/lengthscales, 0)
    return output_scale * np.exp(-0.5 * np.sum(diffs**2, axis=2))

def rbf_covariance2(kernel_params, x, xp):
    output_scale = np.exp(kernel_params[0])
    lengthscales = np.exp(kernel_params[1:])
    K = np.zeros([x.shape[0], xp.shape[0]])
    for i in range(x.shape[0]):
        for j in range(xp.shape[0]):
            K[i][j] = np.exp(-0.5 * np.sum(((x[i,:] - xp[j,:])/lengthscales)**2))
    return output_scale * K


def mvn_logpdf(x, mean, cov):
    log_det_cov = np.linalg.slogdet(cov)[1]
    a = -0.5 * (np.log(2 * np.pi) * len(x) + log_det_cov)
    b = -0.5 * np.dot(np.dot(np.transpose(x - mean), np.linalg.inv(cov)), x - mean)
    return a + b
