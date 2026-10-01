import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# Assuming your backend file is in the same directory and named equations.py
from equations import *

def exact_numerical_threshold(params, delta_val, alpha_bracket=[0.001, 2.0]):
    """
    Calculates the exact lasing threshold by finding the alpha where 
    the maximum real part of the full Jacobian's eigenvalues crosses zero.
    """
    p = params.copy()
    p['delta'] = delta_val
    
    def max_real_eig(alpha):
        p['alpha'] = alpha
        
        # Find exact fixed points including 3D and 4D tensor terms
        fps = fixed_points_num(system, (p,), tolerance=1e-9)
        
        if not fps:
            return -1.0 # Assume stable if no physical fixed points are found
        
        # Track the static equilibrium branch (typically the lowest overall amplitude)
        fp = max(fps, key=lambda x: np.sum(x[:p['N']]))
        
        # Evaluate exact Jacobian at this fixed point, utilizing the nonlinear terms
        J = Jacobian(0.0, fp, p, use_3d=True, use_4d=True, use_optical_coupling=True)
        
        # Return the maximum real part of the eigenvalues
        return np.max(np.real(np.linalg.eigvals(J)))

    try:
        # Find the root where max_real_eig(alpha) == 0 (Hopf bifurcation)
        sol = root_scalar(max_real_eig, bracket=alpha_bracket, method='brentq')
        if sol.converged:
            return sol.root
    except ValueError:
        # Return NaN if the bracket doesn't contain a sign change
        pass
        
    return np.nan

def plot_threshold_comparison(params, delta_range=(-2.0, 4.0), num_points=200):
    """
    Plots the exact numerical threshold against the 0th-order approximation.
    """
    deltas = np.linspace(delta_range[0], delta_range[1], num_points)
    exact_thresholds = []
    
    print("Calculating exact numerical thresholds (this will take a moment)...")
    for d in deltas:
        alpha_c = exact_numerical_threshold(params, d)
        exact_thresholds.append(alpha_c)
        
    # Get 0th-order thresholds using your existing analytical function
    p_0th = params.copy()

    zero_order_threshold = lasing_threshold(p_0th, deltas, return_all=False)

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(deltas, zero_order_threshold, 'k--', label='0th-Order Approximation')
    plt.plot(deltas, exact_thresholds, 'r-', linewidth=2, label=f'Exact Numerical ($\sigma={params["sigma"]}$)')
    
    plt.plot(deltas, upper_boundary(15, deltas), 'b-', linewidth=2, label=f'Bifurcation Boundary')
    plt.plot(deltas, lower_boundary(15, deltas), 'b-', linewidth=2)

    plt.xlabel('Detuning ($\delta$)')
    plt.ylabel('Driving Strength ($\\alpha$)')
    plt.title('Lasing Threshold: Exact vs. 0th-Order Approximation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    y_max = max(np.nanmax(exact_thresholds), np.nanmax(zero_order_threshold))
    plt.ylim(0, y_max * 1.1 if not np.isnan(y_max) else 1.0)
    plt.xlim(delta_range[0], delta_range[1])
    plt.show()

# --- Execution Example ---
# If you have your 'params' dictionary populated with mu, gamma, chi_ijk, chi_ijkl, etc.:
N = 15
M = int(1e3)
use_3d = True
use_4d = True


config_SI = {'N': N,
             'Gammas': np.ones(N)*10,
             'tau': 1/5000,
             'power': 200e-6,
             'detuning': -3e6,
             'd': 10e-9}

config = to_unitless(config_SI)

delta = -2
alpha_bif = upper_boundary(N, delta)
print(f'cusp: {cusp(N)}')
print(f'bifurcation: {alpha_bif}')

config_final = {'N': N,
          'gamma': config['gamma'],
          'mu': mu_spectrum(N),
          'tau': config['tau'],
          'alpha': 0.3,
          'delta': delta,
          'sigma': config['sigma'],
          'xi': np.ones(N),
          'nu': config['nu']}

final_config_SI = to_SI(config_final)
print(final_config_SI)

config_final['alpha'] = 0.01
config_final['delta'] = 4
config_final['sigma'] = 30#config_final['sigma']

config_final['chi_ijk'] = np.load('./tensors/chi_ijk.npy')
config_final['chi_ijkl'] = np.load('./tensors/chi_ijkl.npy')

print(config_final)
plot_threshold_comparison(config_final, delta_range=(-4.0, 4.0))