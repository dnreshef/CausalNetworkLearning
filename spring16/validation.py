from __future__ import print_function
import numpy as np
from simulation import line
from scipy.stats import multivariate_normal as mvn
import GPy

def validate_gpy():
    D = 5
    n_range = range(610, 9, -200)
    n_heldout = 1000
    fixed_x, fixed_y, z = line[0](n_heldout, D)

    for i in xrange(len(n_range)):
        n = n_range[i]
        print("Sample size", n)
        training_x, training_y, z = line[0](n, D)
        y = training_y.flatten()
        kernel = GPy.kern.RBF(D, ARD=True)
        model = GPy.models.GPRegression(training_x, training_y, kernel)
        model.optimize()
        y_pred, sigma2_pred = model.predict(fixed_x)

        model.plot(fixed_inputs=[(i,0) for i in xrange(2, D)])
        plt.show()
        print("MSE: ", np.mean( (fixed_y.flatten()-y_pred.flatten()) ** 2))
        print("Likelihood function:", mvn.logpdf(
            fixed_y.flatten(), y_pred.flatten(), np.diag(sigma2_pred.flatten())))
        print(model)
        print(kernel.lengthscale)

def check_gradient(function_and_gradient, total_dimensions, dimension):
    h = 1e-10
    finite_difference = lambda x: (function_and_gradient(x + h * np.identity(total_dimensions)[dimension])[0] -\
            function_and_gradient(x)[0]) / h
    for x in np.random.rand(100, total_dimensions):
        print(function_and_gradient(x)[1][dimension] - finite_difference(x))
    print()

def test_check_gradient():
    fg = lambda x: (np.linalg.norm(x) ** 2, 2 * x)
    t = 10
    for d in xrange(t):
        check_gradient(fg, t, d)

if __name__ == "__main__":
    test_check_gradient()

