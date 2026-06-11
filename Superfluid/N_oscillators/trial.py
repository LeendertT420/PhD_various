from N_osc_eqs import * 

N = 3
deltas = np.array([-3, 0, 2])

lasing_threshold(N, deltas, t=1, mus=mu_spectrum(N), gs=np.ones(N) * 0.05)