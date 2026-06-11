from equations import * 

N = 3
deltas = np.array([-3, 0, 2])

params = {'N': N,
                   'gamma': np.ones(N) * 0.05,
                   'mu': mu_spectrum(N),
                   'tau': 1.0,
                   'alpha': 1.0,
                   'delta': 0.0,
                   'sigma': 20.0,
                   'chi_ijk': np.load('./tensors/chi_ijk.npy')[:N, :N, :N],
                   'chi_ijkl': np.load('./tensors/chi_ijkl.npy')[:N, :N, :N, :N],
                   'xi': np.ones(N)}

lasing_threshold(params, deltas, verbose=True)
    