from equations import *
import numpy as np
import matplotlib.pyplot as plt

N = 5
deltas = np.linspace(-10, 0, 100)
alphas = np.linspace(0, 10, 100)

low = lower_boundary(N, deltas)
low = low[~np.isnan(low)]
upp = upper_boundary(N, deltas)
deltas_sliced = deltas[~np.isnan(upp)]
upp = upp[~np.isnan(upp)]



d_eff_low = np.array([])
d_eff_upp = np.array([])
x_stars_upp = np.array([])

for d, a_low, a_upp in zip(deltas_sliced, low, upp):
    params = {}
    params['delta'] = d
    params['N'] = N

    params['alpha'] = a_low
    x_star_low = fixed_points_0th_order(params)
    d_eff_low = np.append(d_eff_low, d + N*np.max(x_star_low))

    params['alpha'] = a_upp
    x_star_upp = fixed_points_0th_order(params)
    x_stars_upp = np.append(x_stars_upp, np.max(x_star_upp))
    d_eff_upp = np.append(d_eff_upp, d + N*np.max(x_star_upp))

#plt.plot(d_eff_low, low)
plt.plot(d_eff_upp, upp)
plt.plot(d_eff_upp, -1*(d_eff_upp**2+1)**2/d_eff_upp/2/N)
plt.show()

plt.plot(deltas_sliced, x_stars_upp)
plt.plot(deltas_sliced, (-2/3*deltas_sliced - np.sqrt(deltas_sliced**2-3)/3)/N)
plt.plot(deltas_sliced, 4/3*(-2/3*deltas_sliced + np.sqrt(deltas_sliced**2-3)/3)/N)
plt.plot(deltas_sliced, (-2/3*deltas_sliced + np.sqrt(deltas_sliced**2-3)/3)/N/x_stars_upp)
plt.show()
plt.loglog(-1*deltas_sliced, (-2/3*deltas_sliced + np.sqrt(deltas_sliced**2-3)/3)/N/x_stars_upp)
plt.show()
