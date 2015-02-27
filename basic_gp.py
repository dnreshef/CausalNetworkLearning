import numpy as np
from sklearn import gaussian_process

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
    gp = gaussian_process.GaussianProcess(nugget=1e-6)
    gp.fit(X, y)
    a = np.linalg.norm(gp.predict(X) - y)
    gp.fit(Y, x)
    b = np.linalg.norm(gp.predict(Y) - x)
    return a, b

# Process all data from cause-effect database
with open('results/scores.txt', 'w') as f:
    for i in xrange(1, 89):
        filename = 'pair00%02d.txt' % (i,)
        a, b = gp(*loaddata(filename))

        print filename
        print 'x --> y: %s' % (a,)
        print 'x <-- y: %s' % (b,)
        f.write('%d %s %s\n' % (i, a, b))