import numpy as np
import sys
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
    for i in xrange(10000):
        a = np.random.choice(X)
        b = np.random.choice(X)
        distances.append(abs(a - b))
    distances.sort()
    for i in xrange(q*100, 10000):
        if distances[i] > 0:
            return distances[i]

def filter_outliers(X, Y):
    threshold = 5
    x75, x25 = np.percentile(X, [75 ,25])
    y75, y25 = np.percentile(Y, [75 ,25])
    xiqr = x75 - x25
    yiqr = y75 - y25
    xwhere = np.logical_and(X < x75 + threshold * xiqr, X > x25 - threshold * xiqr)
    ywhere = np.logical_and(Y < y75 + threshold * yiqr, Y > y25 - threshold * yiqr)
    where = np.logical_and(xwhere, ywhere)
    return X[where], Y[where]

def execute(prefix, gpfunc):
    # Process all data from cause-effect database
    with open(prefix + "scores_test.csv", "a") as f:
        f.write("fileno,var_xy,var_yx,MI_xy,MI_yx,NLL_xy,NLL_yx,MIC_xy,MIC_yx,var_xy_norm," +
            "var_yx_norm,MI_xy_norm,MI_yx_norm,NLL_xy_norm,NLL_yx_norm,MIC_xy_norm,MIC_yx_norm\n")
    filenum = 1
    filenum2 = 89
    if len(sys.argv) >= 2:
        filenum = int(sys.argv[1])
        if len(sys.argv) > 2:
            filenum2 = int(sys.argv[2])
        else:
            filenum2 = filenum + 1
    for i in xrange(filenum, filenum2):
        if i == 47:
            print "pair0047.txt is too hard\n"
            continue
        filename = "pair00%02d.txt" % (i,)
        X, Y = loaddata(filename)
        #if X.size > 1000:
        #    print "Size of file %d too big (%d)\n" % (i, X.size)
        #    continue
        if Y.shape[1] > 1:
            print "File %d has multidimensional data\n" % (i,)
            continue
        print "Running GP on %s, %d data points" % (filename, Y.size)
        X_norm, Y_norm = normalize_data(X), normalize_data(Y)
        print np.var(X), np.var(Y)
        scores_xy = gpfunc(X, Y, prefix + "%02dxy" % (i,))
        scores_yx = gpfunc(Y, X, prefix + "%02dyx" % (i,))
        scores_norm_xy = gpfunc(X_norm, Y_norm, prefix + "%02dnormxy" % (i,))
        scores_norm_yx = gpfunc(Y_norm, X_norm, prefix + "%02dnormyx" % (i,))
        output = [i] + [val for pair in zip(scores_xy, scores_yx) for val in pair] +\
        [val for pair in zip(scores_norm_xy, scores_norm_yx) for val in pair]
        with open(prefix + "scores_test.csv", "a") as f:
            f.write("%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % tuple(output))
        print "Garbage collected " + str(gc.collect())