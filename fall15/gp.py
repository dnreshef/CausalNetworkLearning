import numpy as np
import pyGPs
import matplotlib.pyplot as plt

def generate_toy_data():
    x = 2 * np.random.rand(1000, 10) - 1 # 1000 samples, 10-d, uniform in [-1, 1)
    f_x = lambda x: [1 - x[9] ** 2 + 0.1 * np.random.standard_normal()]
    y = np.apply_along_axis(f_x, 1, x)
    z = 2 * np.random.rand(1000, 10) - 1 # 1000 samples, 10-d, uniform in [-1, 1)
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

def run():
    model = pyGPs.GPR()    

    x, y, z = generate_toy_data()
    lel = np.apply_along_axis(np.std, 0, x)
    lengthscale = edistance_at_percentile(x, 50)
#    k = pyGPs.cov.RBFard(log_ell_list=lel, log_sigma=lengthscale) # this doesn't work
    k = pyGPs.cov.Matern(d=7)
    
    model.setPrior(kernel=k) 

    model.optimize(x, y)
    print "Negative log marginal liklihood optimized:", round(model.nlZ,3)

    ymu, ys2, fmu, fs2, lp = model.predict(z)
    ymu = np.reshape(ymu, (ymu.shape[0],))
    ys2 = np.reshape(ymu, (ymu.shape[0],))

    plt.figure()
    plt.plot(z[:,9], ymu, ls='None', marker='+')
    plt.show()
    plt.figure()
    plt.plot(z[:,0], ymu, ls='None', marker='+')
    plt.show()

run()
