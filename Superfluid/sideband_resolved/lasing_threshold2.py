import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.optimize import bisect

# -------------------------------------------------------------------------
# 1. Physics Engine: Solve for Alpha and Detect Existence
# -------------------------------------------------------------------------
def get_critical_alpha(d, n, g):
    """Computes the threshold laser power alpha_c for a specific state."""
    A4 = 2 * g * n**5
    A2 = 2 * n**2 * (g**3 * n + 2 * g**2 * n**2 - g**2 + 2 * g * n**3 - 6 * g * n - 4 * n**2)
    A1 = 2 * d * n**2 * (g + 2 * n)**2
    A0 = 2 * g * n * (g * n + n**2 + 1)**2
    
    roots = np.roots([A4, 0, A2, A1, A0])
    real_roots = roots[np.abs(roots.imag) < 1e-7].real
    
    valid_alphas = []
    for Delta_val in real_roots:
        x_val = Delta_val - d
        alpha_val = x_val * (1 + Delta_val**2)
        if alpha_val > 0:
            valid_alphas.append(alpha_val)
            
    return min(valid_alphas) if valid_alphas else np.nan

def check_limit_cycle_existence(nu_val, d, g):
    """Helper to check if real roots exist at a specific nu."""
    A4 = 2 * g * nu_val**5
    A2 = 2 * nu_val**2 * (g**3 * nu_val + 2 * g**2 * nu_val**2 - g**2 + 2 * g * nu_val**3 - 6 * g * nu_val - 4 * nu_val**2)
    A1 = 2 * d * nu_val**2 * (g + 2 * nu_val)**2
    A0 = 2 * g * nu_val * (g * nu_val + nu_val**2 + 1)**2
    
    roots = np.roots([A4, 0, A2, A1, A0])
    real_roots = roots[np.abs(roots.imag) < 1e-7].real
    
    for Delta_val in real_roots:
        if (Delta_val - d) * (1 + Delta_val**2) > 0:
            return 1
    return -1

def find_critical_nu(d, g):
    """Finds the precise boundary nu_c for a given delta and gamma."""
    nu_min = 0.01
    if check_limit_cycle_existence(nu_min, d, g) == -1:
        return np.nan
        
    nu_max = 20
    while check_limit_cycle_existence(nu_max, d, g) == 1 and nu_max < 50.0:
        nu_max *= 1.5
        
    if check_limit_cycle_existence(nu_max, d, g) == 1:
        return np.nan
        
    try:
        return bisect(check_limit_cycle_existence, nu_min, nu_max, args=(d, g), xtol=1e-4)
    except ValueError:
        return np.nan

# Vectorize alpha calculator for plotting
vec_get_alpha = np.vectorize(get_critical_alpha)

# -------------------------------------------------------------------------
# 2. Logarithmic Precomputation (Executed Once for Fluid Sliders)
# -------------------------------------------------------------------------
print("Precomputing global stability boundary map across logarithmic decades...")
# Generate 50 points spacing evenly on a log scale from 10^-3 to 0.4
gamma_vec = np.logspace(-3, np.log10(0.4), 50)
delta_scan_grid = np.linspace(-5, 5, 100)
absolute_nu_c = []

for g in gamma_vec:
    nu_c_choices = []
    for d in delta_scan_grid:
        nu_val = find_critical_nu(d, g)
        if not np.isnan(nu_val):
            nu_c_choices.append(nu_val)
    absolute_nu_c.append(min(nu_c_choices) if nu_c_choices else np.nan)

print("Precomputation complete. Launching interactive dashboard...")

# -------------------------------------------------------------------------
# 3. Setup Layout & Initial State
# -------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
plt.subplots_adjust(bottom=0.28, wspace=0.25)

init_nu = 0.8
init_gamma = 0.05  # Starting inside the log domain
delta_vec_plot1 = np.linspace(-5, 5, 150)

# -------------------------------------------------------------------------
# 4. Core Render Loop
# -------------------------------------------------------------------------
def render_dashboard(n_curr, g_curr):
    # --- Left Plot: Interactive Lasing Threshold ---
    ax1.clear()
    alpha_values = vec_get_alpha(delta_vec_plot1, n_curr, g_curr)
    
    ax1.plot(delta_vec_plot1, alpha_values, color='royalblue', linewidth=2.5, label=r'$\alpha_c(\delta)$')
    ax1.set_title(f"Lasing Threshold vs Detuning\n(Current: $\\nu={n_curr:.2f}, \\gamma={g_curr:.4f}$)", fontsize=11, fontweight='bold')
    ax1.set_xlabel(r"Laser Detuning $\delta$", fontsize=10)
    ax1.set_ylabel(r"Critical Laser Power $\alpha_c$", fontsize=10)
    ax1.set_xlim(-5, 5)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    if np.all(np.isnan(alpha_values)):
        ax1.text(2.1, 0.5, "No Limit Cycles Possible\n(Cavity Decays Too Fast)", 
                 color='crimson', fontsize=12, ha='center', va='center', weight='bold',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='crimson'))
    else:
        ax1.legend(loc='upper left')

    # --- Right Plot: Absolute Nu Boundary (Logarithmic X-Axis) ---
    ax2.clear()
    ax2.plot(gamma_vec, absolute_nu_c, color='crimson', linewidth=2.5, label=r'Absolute $\nu_{c, \mathrm{min}}$ Ceiling')
    ax2.fill_between(gamma_vec, absolute_nu_c, color='crimson', alpha=0.08, label='Guaranteed Limit Cycle Region')
    
    # Plot tracking dot indicating current configuration space
    ax2.plot(g_curr, n_curr, 'ro', markersize=9, markeredgecolor='k', markeredgewidth=1.5, label='Current State ($\gamma, \nu$)')
    
    ax2.set_title("Absolute Cavity Ceiling (Independent of $\delta$)", fontsize=11, fontweight='bold')
    ax2.set_xlabel(r"Mechanical Damping $\gamma$ (Log Scale)", fontsize=10)
    ax2.set_ylabel(r"Cavity Decay $\nu$", fontsize=10)
    
    # Crucial modifications for the logarithmic axis behavior
    ax2.set_xscale('log')
    ax2.set_xlim(1e-3, 0.4)
    ax2.set_ylim(0, 20)
    ax2.grid(True, which="both", linestyle='--', alpha=0.5)  # Updates grid to capture log decades
    ax2.legend(loc='upper left')

# Initial Draw
render_dashboard(init_nu, init_gamma)

# -------------------------------------------------------------------------
# 5. Interactive Sliders Setup (With Mapped Logarithmic Functionality)
# -------------------------------------------------------------------------
ax_slider_nu    = plt.axes([0.25, 0.12, 0.5, 0.03])
ax_slider_gamma = plt.axes([0.25, 0.05, 0.5, 0.03])

slider_nu = Slider(ax_slider_nu, r'Cavity Decay $\nu$', 0.1, 1.6, valinit=init_nu, valfmt='%1.2f', color='seagreen')

# The gamma slider tracks the exponent from -3.0 (10^-3) to log10(0.4)
slider_gamma = Slider(ax_slider_gamma, r'Mech Damping $\gamma$', -3.0, np.log10(0.4), 
                      valinit=np.log10(init_gamma), color='indianred')

# Override default numeric display to show scientific notation value instead of raw log exponent
slider_gamma.valtext.set_text(f"{init_gamma:.2e}")

def update(val):
    n = slider_nu.val
    
    # Map the linear slider track value back exponentially into the physics engine
    g = 10**slider_gamma.val
    slider_gamma.valtext.set_text(f"{g:.2e}")
    
    render_dashboard(n, g)
    fig.canvas.draw_idle()

slider_nu.on_changed(update)
slider_gamma.on_changed(update)

plt.show()