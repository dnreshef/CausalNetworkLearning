dict(description='Basic GP',
     data_dir='data/',
     n_rand=4,
     skip_complete=False,
     results_dir='/Users/georgedu/Dropbox/Dave and George Shared/gpss_results/',
     iters=100,
     base_kernels='SE,Noise',
     additive_form=True,
     mean='ff.MeanZero()',      # Starting mean
     kernel='ff.NoiseKernel()', # Starting kernel
     lik='ff.LikGauss(sf=-np.Inf)', # Starting likelihood
     starting_subset=750,
     verbose=False,
     search_operators=[('A', ('+', 'A', 'B'), {'A': 'kernel', 'B': 'base'}),
                       ('A', ('*', 'A', 'B'), {'A': 'kernel', 'B': 'base-not-const'}),
                       ('A', 'B', {'A': 'kernel', 'B': 'base'}),
                       ('A', ('None',), {'A': 'kernel'})])
