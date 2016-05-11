# Functions For determining which feature value to set to 0
# and evaluating our model's estimate of the expected outcome change
# when a particular feature is set to 0.
# See example at the bottom of this file for how to use:



from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from simulation import *
from plotting import *
from validation import check_gradient
import GPy
import sys
from PopulationIntervention import *

# Evaluates our population objective when the specified dimension is knocked down.
# knockdown_dim = index of the feature to be knocked down (from 0 to d-1).
def evaluateKnockDown(X, model, knockdown_dim):
    n, d = X.shape
    FIVE_PERCENTILE_ZSCORE = -1.645
    kernel = model.kern
    transformed_pop = X.copy()
    transformed_pop[:, knockdown_dim] = 0.0 # set values according to uniform intervention.    
    all_pts = np.vstack((X, transformed_pop))
    y_mus, y_cov = model.predict(all_pts, full_cov=True, include_likelihood=False)
    normal_mean = np.sum(y_mus[range(n,2*n)] - y_mus[range(n)]) / n
    normal_var = (np.sum(y_cov[n:(2*n),n:(2*n)]) + np.sum(y_cov[0:n,0:n]) - np.sum(y_cov[n:(2*n),0:n]) - np.sum(y_cov[0:n,n:(2*n)])) / (n**2)
    if (normal_var < 0.0) or (np.linalg.norm(transformed_pop - X) < 1e-10):
        normal_var = 0.0
    obj = normal_mean + FIVE_PERCENTILE_ZSCORE * np.sqrt(normal_var)
    return(obj)

# Identifies which knockdown is inferred to bring about 
# the largest INCREASE in expected outcome (Y) with high probability.
# Returns triplet: (best_knockdown, estimated_outcome_change, obj_vals)
# best_knockdown = None if no knockdown with high-probability positive effect was found.
# obj_vals = list of estimated outcome change resulting from knockdown of each dimension.
def findBestKnockDown(X, model):
	d = X.shape[1]
	best_objval = 0.0
	best_knockdown = None
	obj_vals = []
	for feat in range(d):
		objval = evaluateKnockDown(X,model, knockdown_dim=feat)
		obj_vals.append(objval)
		if objval > best_objval:
			best_objval = objval
			best_knockdown = feat
	if best_knockdown is None:
		print("warning: no good knockdown identified")
	return((best_knockdown, best_objval, obj_vals))



# Example Usage:
d = 50; n = 150; noise = 0.2
simulation_func, simulation_eval, true_opt, correct_dims = paraboloid
X, y = simulation_func(n,d,noise)
kernel = GPy.kern.RBF(X.shape[1], ARD=True)
model = GPy.models.GPRegression(X, y, kernel)
model.optimize()
print("ARD lengthscales: ", model.kern.lengthscale)

best_knockdown, best_outcome_change, outcome_changes = findBestKnockDown(X, model)
# paraboloid only depends on last 2 features.

