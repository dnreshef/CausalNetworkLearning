import numpy as np
import sys
import GPy
import pylab
import time
from sklearn.metrics import mutual_info_score

# Calculate mutual information
def calc_MI(x, y):
    bins = np.floor(np.sqrt(len(x)) / 2)
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

# Returns tuple (X, Y) of data parsed from cause-effect database.
def loaddata(filename):
    data = np.loadtxt("data/" + filename)
    X = data[:,:1]
    Y = data[:,1:]
    return X, Y

# Returns an array containing the z-scores of the original data.
def normalize_data(X):
    return (X - np.mean(X)) / np.std(X)

# Computes the q-th percentile for the pairwise euclidean distances in X.
def edistance_at_percentile(X, q):
    distances = []
    for i in xrange(1000):
        a = np.random.choice(X)
        b = np.random.choice(X)
        distances.append(abs(a - b))
    distances.sort()
    return distances[q*10]

# Returns tuple (a, b) of scores, indicating the strength of x --> y causality
# and y --> x causality, respectively.
def gp(X, Y, filename, timing=False):
    a = time.time()

    x = X.ravel()
    y = Y.ravel()
    x_lim = (np.amin(x), np.amax(x))
    y_lim = (np.amin(y), np.amax(y))
    x_model = GPy.models.GPRegression(X, Y)
    y_model = GPy.models.GPRegression(Y, X)
    x_model.kern.lengthscale = edistance_at_percentile(x, 50)
    y_model.kern.lengthscale = edistance_at_percentile(y, 50)

    b = time.time()

    x_model.optimize("bfgs", max_iters=100)

    c = time.time()

    x_model.plot(plot_limits=x_lim)
    pylab.savefig(filename + "xy.png")

    d = time.time()

    y_model.optimize("bfgs", max_iters=100)
    y_model.plot(plot_limits=y_lim)
    pylab.savefig(filename + "yx.png")

    e = time.time()

    y_resid = x_model.predict(X)[0].ravel() - y
    x_resid = y_model.predict(Y)[0].ravel() - x
    var_xy = np.var(y_resid)
    var_yx = np.var(x_resid)
    MI_xy = calc_MI(x, y_resid)
    MI_yx = calc_MI(y, x_resid)
    NLL_xy = -x_model.log_likelihood()
    NLL_yx = -y_model.log_likelihood()

    f = time.time()
    if timing:
        print "setup: %f\noptimize: %f\nplot: %f\nreverse: %f\nscore: %f" %\
        (b-a,c-b,d-c,e-d,f-e)
    return var_xy, var_yx, MI_xy, MI_yx, NLL_xy, NLL_yx

# Process all data from cause-effect database
with open("results/scores_full.csv", "w") as f:
    filenum = 1
    filenum2 = 89
    if len(sys.argv) >= 2:
        filenum = int(sys.argv[1])
        if len(sys.argv) > 2:
            filenum2 = int(sys.argv[2])
        else:
            filenum2 = filenum + 1
    f.write("fileno,var_xy,var_yx,MI_xy,MI_yx,NLL_xy,NLL_yx,var_xy_norm," +
        "var_yx_norm,MI_xy_norm,MI_yx_norm,NLL_xy_norm,NLL_yx_norm\n")
    for i in xrange(filenum, filenum2):
        if i == 47:
            print "pair0047.txt is too hard\n"
            continue
        filename = "pair00%02d.txt" % (i,)
        X, Y = loaddata(filename)
        if X.size > 2000:
            print "Size of file %d too big (%d)\n" % (i, X.size)
            continue
        if Y.shape[1] > 1:
            print "File %d has multidimensional data\n" % (i,)
            continue
        print "Running GP on %s, %d data points" % (filename, Y.size)
        X_norm, Y_norm = normalize_data(X), normalize_data(Y)
        scores = gp(X, Y, "results/%02d" % (i,))
        scores_norm = gp(X_norm, Y_norm, "results/%02dnorm" % (i,))
        print "x --> y: %s %s" % (scores[2], scores_norm[2])
        print "x <-- y: %s %s\n" % (scores[3], scores_norm[3])
        tup = (i,) + scores + scores_norm
        f.write("%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % tup)
