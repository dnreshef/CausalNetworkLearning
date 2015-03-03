import numpy as np
import sys
from sklearn import gaussian_process
from matplotlib import pyplot as pl
from sklearn.metrics import mutual_info_score

# Calculate mutual information
def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

# Returns tuple (X, Y) of data parsed from cause-effect database.
def loaddata(filename):
    data = np.loadtxt('data/' + filename)
    X = data[:,:1]
    Y = data[:,1:]
    return X, Y

# Returns tuple (a, b) of scores, indicating the strength of x --> y causality
# and y --> x causality, respectively.
def gp(X, Y):
    # add some noise
    X = X + np.random.normal(size=X.shape) * 1e-6
    Y = Y + np.random.normal(size=Y.shape) * 1e-6
    x = X.ravel()
    y = Y.ravel()
    x_lin = np.linspace(np.amin(x), np.amax(x), 1000)
    y_lin = np.linspace(np.amin(y), np.amax(y), 1000)
    gp = gaussian_process.GaussianProcess(nugget=1e-6)
    pl.figure()

    gp.fit(X, y)
    y_shade, y_mse = gp.predict(x_lin.reshape((1000, 1)), eval_MSE=True)
    y_pred = gp.predict(X)
    pl.subplot(211)
    pl.plot(x, y, 'bo')
    pl.fill_between(x_lin, y_shade - 1.96 * y_mse, y_shade + 1.96 * y_mse, facecolor='blue', alpha=0.5)

    gp.fit(Y, x)
    x_shade, x_mse = gp.predict(y_lin.reshape((1000, 1)), eval_MSE=True)
    x_pred = gp.predict(Y)
    pl.subplot(212)
    pl.plot(y, x, 'bo')
    pl.fill_between(y_lin, x_shade - 1.96 * x_mse, x_shade + 1.96 * x_mse, facecolor='blue', alpha=0.5)

    pl.show()
    a = calc_MI(x, y_pred - y, 20)
    b = calc_MI(y, x_pred - x, 20)
    return a, b

correct = 0
# Process all data from cause-effect database
with open('results/scores.txt', 'w') as f:
    filenum = 1
    filenum2 = 89
    if len(sys.argv) >= 2:
        filenum = int(sys.argv[1])
        if len(sys.argv) > 2:
            filenum2 = int(sys.argv[2])
        else:
            filenum2 = filenum + 1
    for i in xrange(filenum, filenum2):
        filename = 'pair00%02d.txt' % (i,)
        a, b = gp(*loaddata(filename))

        if a > b:
            correct += 1
        print correct
        print filename
        print 'x --> y: %s' % (a,)
        print 'x <-- y: %s' % (b,)
        f.write('%d %s %s\n' % (i, a, b))
