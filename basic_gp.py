import numpy as np
import sys
import GPy
import pylab
import time
import subprocess
import os
from sklearn.metrics import mutual_info_score

# Constants
prefix = "/Users/georgedu/Dropbox/Dave and George Shared/results/"
timing = False

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
    for i in xrange(10000):
        a = np.random.choice(X)
        b = np.random.choice(X)
        distances.append((a - b)**2)
    distances.sort()
    for i in xrange(q*10, 10000):
        if distances[i] > 0:
            return distances[i]

# Returns tuple (a, b) of scores, indicating the strength of x --> y causality
# and y --> x causality, respectively.
def gp(X, Y, filename=None, bandwidth=50, bfgs_iters=100):
    a = time.time()

    x = X.ravel()
    y = Y.ravel()
    x_lim = (np.amin(x), np.amax(x))
    y_lim = (np.amin(y), np.amax(y))
    xkern = GPy.kern.RBF(1, variance=np.var(y), lengthscale=edistance_at_percentile(x, bandwidth))
    ykern = GPy.kern.RBF(1, variance=np.var(x), lengthscale=edistance_at_percentile(y, bandwidth))
    x_model = GPy.models.GPRegression(X, Y, kernel=xkern)
    y_model = GPy.models.GPRegression(Y, X, kernel=ykern)

    b = time.time()

    x_model.optimize("bfgs", max_iters=bfgs_iters)
    y_model.optimize("bfgs", max_iters=bfgs_iters)

    c = time.time()

    if filename is not None:
        x_model.plot(plot_limits=x_lim)
        pylab.savefig(filename + "xy.png")
        y_model.plot(plot_limits=y_lim)
        pylab.savefig(filename + "yx.png")

    d = time.time()

    y_resid = x_model.predict(X)[0].ravel() - y
    x_resid = y_model.predict(Y)[0].ravel() - x
    var_xy = np.var(y_resid)
    var_yx = np.var(x_resid)
    MI_xy = calc_MI(x, y_resid)
    MI_yx = calc_MI(y, x_resid)
    NLL_xy = -x_model.log_likelihood()
    NLL_yx = -y_model.log_likelihood()
    np.savetxt("data_file.csv", np.hstack((X, y_resid.reshape(y_resid.size, 1), Y, x_resid.reshape(x_resid.size, 1))),
        delimiter=",", header="X,y_resid,Y,x_resid", comments="")
    subprocess.call(["java", "-jar", "MINE_2014_11_10.jar", "data_file.csv", "-adjacentPairs", "exp=0.7", "c=5"])
    with open("DNE,data_file.csv,adjacentpairs,cv=0.0,B=n^0.7,Results.csv", "r") as f:
        f.readline()
        line1 = f.readline().split(",")
        line2 = f.readline().split(",")
        if line1[0] == "X":
            MIC_xy = float(line1[2])
            MIC_yx = float(line2[2])
        else:
            MIC_xy = float(line2[2])
            MIC_yx = float(line1[2])
    os.remove("DNE,data_file.csv,adjacentpairs,cv=0.0,B=n^0.7,Results.csv")
    os.remove("data_file.csv")

    e = time.time()
    if timing:
        print "setup: %f\noptimize: %f\nplot: %f\nscore: %f" %\
        (b-a,c-b,d-c,e-d)
    return var_xy, var_yx, MI_xy, MI_yx, NLL_xy, NLL_yx, MIC_xy, MIC_yx

def gp_bandwidths(X, Y, filename):
    scores = []
    scores.append(gp(X, Y, bandwidth=10, filename=filename + "b10"))
    scores.append(gp(X, Y, bandwidth=50, filename=filename + "b50"))
    scores.append(gp(X, Y, bandwidth=90, filename=filename + "b90"))
    return [min(score_type) for score_type in zip(*scores)]

# Process all data from cause-effect database
with open(prefix + "scores_test.csv", "w") as f:
    filenum = 1
    filenum2 = 89
    if len(sys.argv) >= 2:
        filenum = int(sys.argv[1])
        if len(sys.argv) > 2:
            filenum2 = int(sys.argv[2])
        else:
            filenum2 = filenum + 1
    f.write("fileno,var_xy,var_yx,MI_xy,MI_yx,NLL_xy,NLL_yx,MIC_xy,MIC_yx,var_xy_norm," +
        "var_yx_norm,MI_xy_norm,MI_yx_norm,NLL_xy_norm,NLL_yx_norm,MIC_xy_norm,MIC_yx_norm\n")
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
        scores = gp_bandwidths(X, Y, prefix + "%02d" % (i,))
        scores_norm = gp_bandwidths(X_norm, Y_norm, prefix + "%02dnorm" % (i,))
        print "x --> y (MI): %s %s" % (scores[2], scores_norm[2])
        print "x <-- y (MI): %s %s\n" % (scores[3], scores_norm[3])
        output = [i] + scores + scores_norm
        f.write("%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % tuple(output))
