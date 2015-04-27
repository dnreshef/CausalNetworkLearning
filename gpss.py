import experiment
from gputils import *

exp_params = experiment.load_experiment_details('experiment_details.py')
prefix = exp_params.results_dir

def gpss(X, Y, filename):
    results = experiment.kernel_search_single_step(X, Y, exp_params)
    best_model = results['all_models'][-1]
    print('')
    print('Best BIC = %f' % best_model.bic)
    print('Best model = %s' % best_model.pretty_print())

    #exp_params['mean'] = best_model.mean
    #exp_params['kernel'] = best_model.kernel
    #exp_params['lik'] = best_model.likelihood

execute(prefix, gpss)