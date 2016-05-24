# Functions For determining which feature value to set to 0
# and evaluating our model's estimate of the expected outcome change
# when a particular feature is set to 0.
# See example at the bottom of this file for how to use.

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
from scipy import stats
from scipy import linalg
import statsmodels.sandbox.stats.multicomp as mcp
import sklearn.linear_model as lm

# Evaluates our population objective when the specified dimension is knocked down.
# knockdown_dim = index of the feature to be knocked down (from 0 to d-1).
def evaluateKnockDown(X, model, knockdown_dim, z, zeros):
    n, d = X.shape
    FIVE_PERCENTILE_ZSCORE = z
    kernel = model.kern
    transformed_pop = X.copy()
    transformed_pop[:, knockdown_dim] = zeros[knockdown_dim] #0.0 # set values according to uniform intervention.    
    all_pts = np.vstack((X, transformed_pop))
    y_mus, y_cov = model.predict(all_pts, full_cov=True, include_likelihood=False)
    normal_mean = np.sum(y_mus[range(n,2*n)] - y_mus[range(n)]) / n
    normal_var = (np.sum(y_cov[n:(2*n),n:(2*n)]) + np.sum(y_cov[0:n,0:n]) - np.sum(y_cov[n:(2*n),0:n]) - np.sum(y_cov[0:n,n:(2*n)])) / (n**2)
    if (normal_var < 0.0) or (np.linalg.norm(transformed_pop - X) < 1e-10):
        normal_var = 0.0
    # print("Mean:", normal_mean, "Rest:", FIVE_PERCENTILE_ZSCORE * np.sqrt(normal_var))
    obj = normal_mean + FIVE_PERCENTILE_ZSCORE * np.sqrt(normal_var)
    return(obj)

# Identifies which knockdown is inferred to bring about 
# the largest INCREASE in expected outcome (Y) with high probability.
# Returns triplet: (best_knockdown, estimated_outcome_change, obj_vals)
# best_knockdown = None if no knockdown with high-probability positive effect was found.
# obj_vals = list of estimated outcome change resulting from knockdown of each dimension.
def findBestKnockDown(X, model, zscore, knockdownZeros):
	d = X.shape[1]
	best_objval = 0.0
	best_knockdown = None
	obj_vals = []
	for feat in range(d):
		objval = evaluateKnockDown(X,model, knockdown_dim=feat, z=zscore, zeros=knockdownZeros)
		obj_vals.append(objval)
		if objval > best_objval:
			best_objval = objval
			best_knockdown = feat
	if best_knockdown is None:
		print("warning (zscore=", zscore, "): no good knockdown identified")
	return((best_knockdown, best_objval, obj_vals))

# Example Usage:
# d = 50; n = 150; noise = 0.2
# simulation_func, simulation_eval, true_opt, correct_dims = paraboloid
# X, y = simulation_func(n,d,noise)
# print("X shape:", X.shape)
# print("y shape:", y.shape)
# kernel = GPy.kern.RBF(X.shape[1], ARD=True)
# model = GPy.models.GPRegression(X, y, kernel)
# model.optimize()
# print("ARD lengthscales: ", model.kern.lengthscale)

# best_knockdown, best_outcome_change, outcome_changes = findBestKnockDown(X, model)
# print(best_knockdown)
# print(outcome_changes)
# paraboloid only depends on last 2 features.




# Rescale an array to [0,1]
def rescale(X,ms):
    for xc in range(X.shape[1]):
        colMin = X[:,xc].min()
        colMax = X[:,xc].max()
        X[:,xc] = (X[:,xc] - colMin) / (colMax-colMin)
        ms[xc] = (ms[xc] - colMin) / (colMax-colMin)
    return (X,ms)

# Center an array
def center(y):
    return (y - np.mean(y))

# Get the list of expression ratios (M) for the expression of each gene when it is knocked out iteself (to pulg in as the 0 values in evaluateKnockDown)
def getInterventionMsForXs(yeastData):
    interventionData_MsForKnockouts = np.genfromtxt('/Users/Dave/Dropbox/Desktop/PNAS Data Version/InterventionData_MCols_Ys.txt', delimiter='\t', dtype=None)
    ms = []
    for xi in range(len(yeastData[0,1:-1])):
        xRow = interventionData_MsForKnockouts[:,0].tolist().index(yeastData[0,1+xi])
        xCol = interventionData_MsForKnockouts[0,:].tolist().index(yeastData[0,1+xi])
        ms.append(interventionData_MsForKnockouts[xRow, xCol].astype(float))
    return ms

# Analysis using independent marginal linear regressions. Fit a separate linear regression to each predictor gene x_g separately.
# Choose gene x_g who's Pearson correlation is significant and which has the largest Pearson.  If none is significant, do nothing.
# P-values are adjusted for multiple testing since we're fitting separate regression models to each of the TFs.
# REMEMBER: adjust which Pearson value you choose (max or min) depending on whether Y is being up/down-regulated!
def findBestKnockdownUsingMarginalLinearRegression(X, y, knockdownZeros):
    d = X.shape[1]
    best_objval = 0.0
    best_knockdown = None
    obj_vals = []
    p_vals = []
    for feat in range(d):
        #objval, pVal = stats.pearsonr(X[:,feat], y[:,0])
        objval, intercept, r_value, pVal, std_err = stats.linregress(X[:,feat], y[:,0])
        obj_vals.append(objval * (knockdownZeros[feat]-np.mean(X[:,feat])))
        p_vals.append(pVal)
    #print("Coeffs:", obj_vals)
    #print("Pvals:", p_vals)

    rejected, p_vals_corrected = mcp.fdrcorrection0(p_vals, alpha=0.05)
    #print("Corrected:", p_vals_corrected)
    #print("Rejected:", rejected)

    for feat in range(d):
        if rejected[feat] == True and obj_vals[feat] > best_objval:
            best_objval = obj_vals[feat]
            best_knockdown = feat
        if rejected[feat] == False:
            obj_vals[feat] = 0
    if best_knockdown is None:
        print("warning (indep. marg. regr.): no good knockdown identified")
    return((best_knockdown, best_objval, obj_vals))


# Analysis using multivariate linear regressions. Fit multivariate linear regression to all predictor genes.
# Choose gene x_g who's coefficient beta is significant and which has the largest (or smallest) beta*(0-mean(gene)).
# If none is significant, do nothing.
# REMEMBER: adjust which value you choose (max or min) depending on whether Y is being up/down-regulated!
def findBestKnockdownUsingMultipleLinearRegression(X, y, knockdownZeros):
    d = X.shape[1]
    best_objval = 0.0
    best_knockdown = None
    obj_vals = []
    p_vals = []
    model = lm.LinearRegression()
    model.fit(X, y[:,0])
    #print("Coeffs:", model.coef_)
    obj_vals = model.coef_ * (knockdownZeros-np.mean(X, axis=0))
    #print("Obj vals:", obj_vals)

    n, k = X.shape
    yHat = np.matrix(model.predict(X)).T

    # Change X and Y into numpy matricies. X also has a column of ones added to it.
    x = np.hstack((np.ones((n,1)),np.matrix(X)))
    y = np.matrix(y).T

    # Degrees of freedom.
    df = float(n-k-1)

    # Sample variance.     
    sse = np.sum(np.square(yHat - y),axis=0)
    sampleVariance = sse/df

    # Sample variance for x.
    sampleVarianceX = x.T*x

    # Covariance Matrix = [(s^2)(X'X)^-1]^0.5. (sqrtm = matrix square root.  ugly)
    covarianceMatrix = linalg.sqrtm(sampleVariance[0,0]*sampleVarianceX.I)

    # Standard erros for the difference coefficients: the diagonal elements of the covariance matrix.
    se = covarianceMatrix.diagonal()[1:]

    # T statistic for each beta.
    betasTStat = np.zeros(len(se))
    for i in xrange(len(se)):
        betasTStat[i] = model.coef_[i]/se[i]

    # P-value for each beta. This is a two sided t-test, since the betas can be 
    # positive or negative.
    betasPValue = 1 - stats.t.cdf(abs(betasTStat),df)
    #print("Betas pvals:", betasPValue)

    for feat in range(d):
        if betasPValue[feat] < 0.05 and obj_vals[feat] > best_objval:
            best_objval = obj_vals[feat]
            best_knockdown = feat
        if betasPValue[feat] >= 0.05:
            obj_vals[feat] = 0
    if best_knockdown is None:
        print("warning (multi. regr.): no good knockdown identified")
    return((best_knockdown, best_objval, obj_vals))




# Yeast Gene Perturbation, iterate over all possible Ys
yCandidates = np.genfromtxt('/Users/Dave/Dropbox/Desktop/PNAS Data Version/finalYcandidates.txt', delimiter='\t', dtype=None)
best_kockdown_genes = np.empty([len(yCandidates), 4], dtype='|S12')
allMethodsOutcomes = []

# Uncomment for 'commonlyAffectedYCandidates.txt'
# yCandidates = np.array([ [y[0], y[1]] for y in yCandidates])

for yi,Y in enumerate(yCandidates):
    if yi < len(yCandidates):
        print("=========== Y number:", yi, "(", yCandidates[yi,0], ")===========")
        inputFile = 'WTData_X=gene-specific_transcription_factor_Y=' + Y[0] + '.txt'
        yeastData = np.genfromtxt('/Users/Dave/Dropbox/Desktop/PNAS Data Version/TFs_Y=smallMolMet/'+inputFile, delimiter='\t', dtype=None)
        X = yeastData[1:,1:-1].astype(float)
        interventionMsForXs = getInterventionMsForXs(yeastData)
        X,scaledInterventionMsForXs = rescale(X.astype(float), interventionMsForXs)
        y = np.empty([len(yeastData)-1, 1])
        y[:,0] = center(-1*yeastData[1:,-1].astype(float))
        #print(X)
        #print(y)
        #print("X shape:", X.shape)
        #print("y shape:", y.shape)

        # Our model
        kernel = GPy.kern.RBF(X.shape[1], ARD=True)
        model = GPy.models.GPRegression(X, y, kernel)
        model.optimize()
        print("ARD lengthscales: ", model.kern.lengthscale)
        best_knockdown, best_outcome_change, outcome_changes = findBestKnockDown(X, model, -1.645, scaledInterventionMsForXs)
        if best_knockdown is None:
            best_kockdown_genes[yi, 0] = "None"
        else:
            best_kockdown_genes[yi, 0] = yeastData[0, best_knockdown+1].astype(str)
        allMethodsOutcomes.append(outcome_changes)
        print("Best knockdown:", best_knockdown)#, "(", yeastData[0, best_knockdown+1].astype(str), ")")
        print("Best outcome change:", best_outcome_change)
        print("Outcome changes:", outcome_changes)
        print()

        # Out model, no uncertainty
        best_knockdown, best_outcome_change, outcome_changes = findBestKnockDown(X, model, 0, scaledInterventionMsForXs)
        if best_knockdown is None:
        best_kockdown_genes[yi, 1] = "None"
        else:
            best_kockdown_genes[yi, 1] = yeastData[0, best_knockdown+1].astype(str)
        allMethodsOutcomes.append(outcome_changes)

        # Marginal linear regression
        best_knockdown, best_outcome_change, outcome_changes = findBestKnockdownUsingMarginalLinearRegression(X, y, scaledInterventionMsForXs)

        if best_knockdown is None:
            best_kockdown_genes[yi, 2] = "None"
        else:
            best_kockdown_genes[yi, 2] = yeastData[0, best_knockdown+1].astype(str)
        allMethodsOutcomes.append(outcome_changes)
        print("MARGINAL REGRESSION")
        print("Best knockdown:", best_knockdown)#, "(", yeastData[0, best_knockdown+1].astype(str), ")")
        print("Best outcome change:", best_outcome_change)
        print("Outcome changes:", outcome_changes)
        print()
        
        # Multivariate linear regression
        best_knockdown, best_outcome_change, outcome_changes = findBestKnockdownUsingMultipleLinearRegression(X, y, scaledInterventionMsForXs)

        if best_knockdown is None:
            best_kockdown_genes[yi, 3] = "None"
        else:
            best_kockdown_genes[yi, 3] = yeastData[0, best_knockdown+1].astype(str)
        allMethodsOutcomes.append(outcome_changes)
        print("MULTIVAR REGRESSION")
        print("Best knockdown:", best_knockdown)#, "(", yeastData[0, best_knockdown+1].astype(str), ")")
        print("Best outcome change:", best_outcome_change)
        print("Outcome changes:", outcome_changes)
        print()

        # Output
        np.savetxt('TFs_Y=smallMolMet/outcomeChanges'+inputFile.split('Data')[1], np.transpose(allMethodsOutcomes), delimiter='\t', fmt="%s")
        #print()
        #print()

interventionData = np.genfromtxt('/Users/Dave/Dropbox/Desktop/PNAS Data Version/InterventionData_MCols.txt', delimiter='\t', dtype=None)
yEffects = np.copy(interventionData[:len(yCandidates), :11])
yEffects[:,1:] = np.zeros([len(yEffects), len(yEffects[0])-1])
for i in range(len(yCandidates)):
    if i < len(yCandidates):
        print("=========== I number:", i, "===========")
        yEffects[i,0] = yCandidates[i,0]
        #print("Y cand:", yCandidates[i,0])
        row = np.where(interventionData[:,0] == yCandidates[i,0])[0][0]
        #print("Row:", i, row, min(interventionData[row,1:].astype(float)))
        yEffects[i,1] = interventionData[ 0, np.argmin(interventionData[row,1:].astype(float))+1 ]
        yEffects[i,2] = min(interventionData[row,1:].astype(float))
        #print("Min:",yEffects[i,1], yEffects[i,2])
        
        # Our method
        yEffects[i,3] = best_kockdown_genes[i,0]
        #print("Best knockdown:", best_kockdown_genes[i,0])
        #print(np.where(interventionData[0,:] == best_kockdown_genes[i,0]))
        if yEffects[i,3] == "None":
            yEffects[i,4] = 0
            #print(yEffects[i,4])
        else:
            col = np.where(interventionData[0,:] == best_kockdown_genes[i,0])[0][0]
            #print("Col:",col)
            yEffects[i,4] = interventionData[ row, col ]

        # Our method, no uncertainty
        yEffects[i,5] = best_kockdown_genes[i,1]
        #print("Best knockdown:", best_kockdown_genes[i,1])
        #print(np.where(interventionData[0,:] == best_kockdown_genes[i,1]))
        if yEffects[i,5] == "None":
            yEffects[i,6] = 0
            #print(yEffects[i,6])
        else:
            col = np.where(interventionData[0,:] == best_kockdown_genes[i,1])[0][0]
            #print("Col:",col)
            yEffects[i,6] = interventionData[ row, col ]

        # Marginal linear regression
        yEffects[i,7] = best_kockdown_genes[i,2]
        #print("Best knockdown:", best_kockdown_genes[i,2])
        #print(np.where(interventionData[0,:] == best_kockdown_genes[i,2]))
        if yEffects[i,7] == "None":
            yEffects[i,8] = 0
            #print(yEffects[i,8])
        else:
            col = np.where(interventionData[0,:] == best_kockdown_genes[i,2])[0][0]
            #print("Col:",col)
            yEffects[i,8] = interventionData[ row, col ]

        # Multivariate linear regression
        yEffects[i,9] = best_kockdown_genes[i,3]
        #print("Best knockdown:", best_kockdown_genes[i,3])
        #print(np.where(interventionData[0,:] == best_kockdown_genes[i,3]))
        if yEffects[i,9] == "None":
            yEffects[i,10] = 0
            #print(yEffects[i,10])
        else:
            col = np.where(interventionData[0,:] == best_kockdown_genes[i,3])[0][0]
            #print("Col:",col)
            yEffects[i,10] = interventionData[ row, col ]

results = yEffects[yEffects[:,2].astype(float).argsort()[::-1]]
results = np.vstack( (["YCandidate", "OptX", "OptXEffect", "OurModelX", "OurModelXEffect", "NoUncert.X", "NoUncert.XEffect", "MargRegX", "MargRegXEffect", "MultiRegX", "MultiRegXEffect"], results) )
np.savetxt('TFs_Y=smallMolMet/resultsForAllY_downreg.txt', results, delimiter='\t', fmt="%s")