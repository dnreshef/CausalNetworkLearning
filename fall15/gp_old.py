import numpy as np
import pyGPs
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs

def generate_toy_data():
    x = 2 * np.random.rand(100, 2) - 1 # 1000 samples, 10-d, uniform in [-1, 1)
    f_x = lambda x: [1 - x[-1] ** 2 + 0.01 * np.random.standard_normal()]
    y = np.apply_along_axis(f_x, 1, x)
    z = 2 * np.random.rand(100, 2) - 1 # 1000 samples, 10-d, uniform in [-1, 1)
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
    t = np.hstack((x, top, bottom))
    return t[t[:,0].argsort()]

def run():
    model = pyGPs.GPR()    

    x, y, z = generate_toy_data()
    lel = np.apply_along_axis(np.std, 0, x)
    print "parameters"
    print lel
    lengthscale = edistance_at_percentile(x, 50)
    print lengthscale
    # TODO: set non-default parameters
    k = pyGPs.cov.RBFard(log_ell_list=[0.01,0.01],log_sigma=0.01)#D=x.shape[1])
    m = pyGPs.mean.Const()

    model.setPrior(mean=m,kernel=k)

    print "hyperparameters"
    print k.hyp
    model.optimize(x, y)
    print "posterior", model.posterior
    print "Negative log marginal liklihood optimized:", round(model.nlZ,3)

    def objective(x):
        ymu, ys2, fmu, fs2, lp = model.predict(x.reshape((1, len(x))))
        ret = ymu - 1.645 * np.sqrt(ys2)
        return ret[0][0]

    x_opt = fmin_bfgs(lambda x: objective(x) * -1, np.arange(0, 0.2, 0.1))
    print "Optimized value of x:", x_opt

    ymu, ys2, fmu, fs2, lp = model.predict(z)
    q_95 = ymu + 1.645 * np.sqrt(ys2)
    q_5 = ymu - 1.645 * np.sqrt(ys2)
    t1 = sort_for_plotting(z[:,-1].reshape(len(z), 1), q_95, q_5)
    t2 = sort_for_plotting(z[:,0].reshape(len(z), 1), q_95, q_5)

    plt.figure()
    ymu = np.reshape(ymu, (ymu.shape[0],))
    plt.plot(z[:,-1], ymu, ls='None', marker='+')
    plt.fill_between(t1[:,0], t1[:,1], t1[:,2], facecolor=[0.7539, 0.89453125, 0.62890625, 1.0], linewidths=0)
    plt.show()
    plt.figure()
    plt.plot(z[:,0], ymu, ls='None', marker='+')
    plt.fill_between(t2[:,0], t2[:,1], t2[:,2], facecolor=[0.7539, 0.89453125, 0.62890625, 1.0], linewidths=0)
    plt.show()

run()
