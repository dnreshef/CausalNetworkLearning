import numpy as np
import GPy
import pylab
import time
import subprocess
import os
from gputils import *

# Constants
prefix = "/Users/georgedu/Dropbox/Dave and George Shared/results/"
timing = False

# Returns tuple (a, b) of scores, indicating the strength of x --> y causality
# and y --> x causality, respectively.
def gp(X, Y, filename=None, bandwidth=50, bfgs_iters=100):
    a = time.time()

    #X, Y = filter_outliers(X, Y)
    #X = X.reshape((X.size, 1))
    #Y = Y.reshape((Y.size, 1))
    x = X.ravel()
    y = Y.ravel()
    x_lim = (np.amin(x), np.amax(x))
    lengthscale = edistance_at_percentile(x, bandwidth)
    xkern = GPy.kern.RBF(1, variance=1., lengthscale=lengthscale)

    x_optimize = GPy.models.GPRegression(X, Y, kernel=xkern)

    b = time.time()

    # No optimize restart
    x_optimize.optimize(max_iters=bfgs_iters)
    x_model = x_optimize

    c = time.time()

    if filename is not None:
        x_model.plot(plot_limits=x_lim)
        pylab.savefig(filename)

    d = time.time()

    y_resid = x_model.predict(X)[0].ravel() - y
    var_xy = np.var(y_resid)
    MI_xy = calc_MI(x, y_resid)
    NLL_xy = -x_model.log_likelihood()
    np.savetxt("data_file.csv", np.hstack((X, y_resid.reshape(y_resid.size, 1))),
        delimiter=",", header="X,y_resid", comments="")
    subprocess.call(["java", "-jar", "MINE_2014_11_10.jar", "data_file.csv", "-adjacentPairs", "exp=0.7", "c=5"])
    with open("DNE,data_file.csv,adjacentpairs,cv=0.0,B=n^0.7,Results.csv", "r") as f:
        f.readline()
        line1 = f.readline().split(",")
        MIC_xy = float(line1[2])
    os.remove("DNE,data_file.csv,adjacentpairs,cv=0.0,B=n^0.7,Results.csv")
    os.remove("data_file.csv")

    e = time.time()
    if timing:
        print "setup: %f\noptimize: %f\nplot: %f\nscore: %f" %\
        (b-a,c-b,d-c,e-d)
    return var_xy, MI_xy, NLL_xy, MIC_xy

def gp_bandwidths(X, Y, filename):
    scores = []
    scores.append(gp(X, Y, bandwidth=10, filename=filename + "b10"))
    scores.append(gp(X, Y, bandwidth=50, filename=filename + "b50"))
    scores.append(gp(X, Y, bandwidth=90, filename=filename + "b90"))
    return [min(score_type) for score_type in zip(*scores)]

execute(prefix, gp_bandwidths)
