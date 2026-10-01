import numpy as np
import matplotlib.pyplot as plt
from equations import *
from tqdm import tqdm

N = 15
use_3d = True
use_4d = True

config_SI = {
    'N': N,
    'Gammas': np.ones(N) * 10,
    'tau': 1 / 5000,
    'power': 80e-6,
    'detuning': -15.5e6,
    'd': 10e-9
}

config = to_unitless(config_SI)

delta = -4
alpha_bif = upper_boundary(N, delta)

config = {
    'N': N,
    'gamma': config['gamma'],
    'mu': mu_spectrum(N),
    'tau': config['tau'],
    'alpha': alpha_bif - 0.1,
    'delta': delta,
    'sigma': config['sigma'],
    'chi_ijk': np.load('./tensors/chi_ijk.npy')[:N, :N, :N],
    'chi_ijkl': np.load('./tensors/chi_ijkl.npy')[:N, :N, :N, :N],
    'xi': np.ones(N),
    'nu': config['nu']
}

deltas = np.linspace(-4, -np.sqrt(3), 400)
alphas = np.linspace(0.1, 1, 900)

N_fixed_points = np.zeros((len(alphas), len(deltas)))

for i, d in tqdm(enumerate(deltas)):
    config['delta'] = d
    for j, a in enumerate(alphas):
        config['alpha'] = a
        fixed_points = fixed_points_num(config, num_tries=100)
        N_fixed_points[j, i] = len(fixed_points)  # Fixed indexing: [j, i] for (alpha, delta) grid


# --- FIRST-ORDER BOUNDARY CORRECTION IMPLEMENTATION ---

def boundary_first_order(N, deltas, chi_ijk, sigma, branch='upper'):
    """
    Calculates the first-order perturbation correction to the bifurcation boundary alpha_c(delta).
    """
    epsilon = 1.0 / sigma
    bar_Lambda = np.sum(chi_ijk)  # Sum of all chi_ijk over i, j, k
    
    # Solve quadratic equation for critical x_bar_c: 3N^2 x^2 + 4 N delta x + (delta^2 + 1) = 0
    # Roots: (-2*delta ± sqrt(delta^2 - 3)) / (3*N)
    discriminant = deltas**2 - 3.0
    discriminant = np.maximum(discriminant, 0)  # Guard against float precision below sqrt(3)
    
    if branch == 'upper':
        x_bar_c = (-2.0 * deltas + np.sqrt(discriminant)) / (3.0 * N)
    else:  # 'lower'
        x_bar_c = (-2.0 * deltas - np.sqrt(discriminant)) / (3.0 * N)
        
    # Zeroth order alpha_c
    alpha_0 = x_bar_c * ((N * x_bar_c + deltas)**2 + 1.0)
    
    # First order correction: alpha_c^(1) = - bar_Lambda * x_bar_c^2 * ((N * x_bar_c + deltas)^2 + 1)
    alpha_1 = - bar_Lambda * (x_bar_c**2) * ((N * x_bar_c + deltas)**2 + 1.0)
    
    return alpha_0 + epsilon * alpha_1


# Compute first-order boundaries
upper_1st = boundary_first_order(N, deltas, config['chi_ijk'], config['sigma'], branch='upper')
lower_1st = boundary_first_order(N, deltas, config['chi_ijk'], config['sigma'], branch='lower')

# --- PLOTTING & ACCURACY CHECK ---

plt.figure(figsize=(10, 6))

# Numerical count heatmap
plt.pcolormesh(deltas, alphas, N_fixed_points, shading='nearest', cmap='viridis')

# Zeroth-order boundaries (Black lines)
plt.plot(deltas, upper_boundary(N, deltas), 'k--', label='0th Order Boundary (Uncorrected)')
plt.plot(deltas, lower_boundary(N, deltas), 'k--')

# First-order boundaries (Red dashed lines)
plt.plot(deltas, upper_1st, 'r-', linewidth=2, label='1st Order Boundary (Corrected)')
plt.plot(deltas, lower_1st, 'r-', linewidth=2)

plt.xlim(np.min(deltas), np.max(deltas))
plt.ylim(np.min(alphas), np.max(alphas))
plt.xlabel(r'$\delta$')
plt.ylabel(r'$\alpha$')
plt.title('Bifurcation Diagram: Numerical vs 0th and 1st Order Boundaries')
plt.legend(loc='best')
plt.colorbar(label='Number of Fixed Points')
plt.tight_layout()
plt.show()