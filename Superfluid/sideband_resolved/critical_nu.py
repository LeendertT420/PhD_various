import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect

# -------------------------------------------------------------------------
# 1. Physics Engine: Boundary Detector
# -------------------------------------------------------------------------
def check_limit_cycle_existence(nu_val, d, g):
    """
    Returns 1 if a physical Hopf bifurcation threshold exists at this nu,
    and -1 if the threshold has vanished into complex-root territory.
    """
    # Calculate the quartic coefficients for the given parameter set
    A4 = 2 * g * nu_val**5
    A2 = 2 * nu_val**2 * (g**3 * nu_val + 2 * g**2 * nu_val**2 - g**2 + 2 * g * nu_val**3 - 6 * g * nu_val - 4 * nu_val**2)
    A1 = 2 * d * nu_val**2 * (g + 2 * nu_val)**2
    A0 = 2 * g * nu_val * (g * nu_val + nu_val**2 + 1)**2
    
    # Extract polynomial roots
    roots = np.roots([A4, 0, A2, A1, A0])
    real_roots = roots[np.abs(roots.imag) < 1e-7].real
    
    # Verify if any real root generates a physically valid positive laser power
    for Delta_val in real_roots:
        x_val = Delta_val - d
        alpha_val = x_val * (1 + Delta_val**2)
        if alpha_val > 0:
            return 1  # Limit cycles can exist here!
            
    return -1  # Adiabatic regime; limit cycles are dead.

# -------------------------------------------------------------------------
# 2. Root Finder: Locate Exact Critical Nu Crossings
# -------------------------------------------------------------------------
def find_critical_nu(d, g):
    """
    Finds the exact critical cavity decay (nu_c) where the system transitions
    from supporting limit cycles to absolute stability.
    """
    nu_min = 0.01
    
    # If limit cycles don't even exist at ultra-low cavity decay, skip it
    if check_limit_cycle_existence(nu_min, d, g) == -1:
        return np.nan
        
    # Dynamically expand the upper bound to safely bracket the root
    nu_max = 2.0
    while check_limit_cycle_existence(nu_max, d, g) == 1 and nu_max < 50.0:
        nu_max *= 1.5
        
    # If it still exists at nu = 50, the threshold is outside our search window
    if check_limit_cycle_existence(nu_max, d, g) == 1:
        return np.nan
        
    # Use high-precision numerical bisection to pin down the exact crossing point
    try:
        return bisect(check_limit_cycle_existence, nu_min, nu_max, args=(d, g), xtol=1e-4)
    except ValueError:
        return np.nan

# -------------------------------------------------------------------------
# 3. Generate the Parametric Coordinate Grid
# -------------------------------------------------------------------------
print("Computing the critical cavity decay surface (this may take a few seconds)...")

# Define meaningful experimental bounds for delta and gamma
delta_vec = np.linspace(-4, 4.0, 200)
gamma_vec = np.linspace(0.01, 0.1, 200)
delta_grid, gamma_grid = np.meshgrid(delta_vec, gamma_vec)

# Initialize the critical height array
nu_c_surface = np.zeros_like(delta_grid)

# Map the grid row by row
for i in range(len(gamma_vec)):
    for j in range(len(delta_vec)):
        nu_c_surface[i, j] = find_critical_nu(delta_grid[i, j], gamma_grid[i, j])

print("Computation complete. Generating 3D Plot...")

# -------------------------------------------------------------------------
# 4. 3D Surface Visualization
# -------------------------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Mask out any parameter combinations where no boundary exists
masked_nu_c = np.ma.masked_where(np.isnan(nu_c_surface), nu_c_surface)

# Render the landscape
surf = ax.plot_surface(delta_grid, gamma_grid, masked_nu_c, cmap='viridis',
                        edgecolor='none', alpha=0.9)

# Add a colorbar to read values easily
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
cbar.set_label(r'Critical Cavity Decay ($\nu_c$)', fontsize=11)

# Labels and Presentation
ax.set_title(r"Phase Boundary: Maximum Cavity Decay $\nu_c$ for Limit Cycles", fontsize=13, pad=15, fontweight='bold')
ax.set_xlabel(r"Laser Detuning $\delta$", fontsize=11)
ax.set_ylabel(r"Mechanical Damping $\gamma$", fontsize=11)
ax.set_zlabel(r"Critical Decay $\nu_c$", fontsize=11)

# Adjust the camera perspective to clearly see the slope trends
ax.view_init(elev=25, azim=-45)

plt.tight_layout()
plt.show()