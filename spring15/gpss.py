import experiment
import subprocess
import os
import gc
import pylab
from gputils import *

prefix = experiment.load_experiment_details('experiment_details.py')['results_dir']

def gpss(X, Y, filename):
    exp_params = experiment.load_experiment_details('experiment_details.py')
    results = experiment.kernel_search_single_step(X, Y, exp_params)
    best_model = results['all_models'][-1]
    x_lim = (np.amin(X.ravel()), np.amax(X.ravel()))
    print('')
    print('Best BIC = %f' % best_model.bic)
    print('Best model = %s' % best_model.pretty_print())

    #exp_params['mean'] = best_model.mean
    #exp_params['kernel'] = best_model.kernel
    #exp_params['lik'] = best_model.likelihood

    if filename is not None:
        x_model = best_model.create_gpy_model(X, Y)
        x_model.plot(plot_limits=x_lim)
        pylab.savefig(filename)
        best_model.gpy_model = None
    y_resid = (best_model.gpy_predict(X, Y, X)['mean'] - Y)
    var_xy = np.var(y_resid.ravel())
    MI_xy = calc_MI(X.ravel(), y_resid.ravel())
    NLL_xy = 0 #todo
    np.savetxt("gpss_data.csv", np.hstack((X, y_resid)),
        delimiter=",", header="X,y_resid", comments="")
    subprocess.call(["java", "-jar", "MINE_2014_11_10.jar", "gpss_data.csv", "-adjacentPairs", "exp=0.7", "c=5"])
    with open("DNE,gpss_data.csv,adjacentpairs,cv=0.0,B=n^0.7,Results.csv", "r") as f:
        f.readline()
        line1 = f.readline().split(",")
        MIC_xy = float(line1[2])
    os.remove("DNE,gpss_data.csv,adjacentpairs,cv=0.0,B=n^0.7,Results.csv")
    os.remove("gpss_data.csv")

    gc.collect()
    return var_xy, MI_xy, NLL_xy, MIC_xy

execute(prefix, gpss)
