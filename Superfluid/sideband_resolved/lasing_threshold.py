import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# -------------------------------------------------------------------------
# 1. Physics Engine: Solve Routh-Hurwitz Quartic for Critical Alpha
# -------------------------------------------------------------------------
def get_critical_alpha(d, n, g):
    """
    Computes the minimum physical laser power (alpha_c) where a Hopf 
    bifurcation occurs, given detuning (d), decay (n), and damping (g).
    """
    # Quartic coefficients derived from the Routh-Hurwitz boundary condition
    A4 = g * n**4
    A2 = n * (g**3 * n + 2 * g**2 * n**2 - g**2 + 2 * g * n**3 - 6 * g * n - 4 * n**2)
    A1 = d * n * (g + 2 * n)**2
    A0 = g * (g * n + n**2 + 1)**2
    
    # Solve for the effective detuning (Delta)
    roots = np.roots([A4, 0, A2, A1, A0])
    
    # Filter for purely real roots
    real_roots = roots[np.abs(roots.imag) < 1e-7].real
    
    valid_alphas = []
    for Delta_val in real_roots:
        x_val = Delta_val - d
        alpha_val = x_val * (1 + Delta_val**2)
        if alpha_val > 0:  # Alpha must be positive to be physically meaningful
            valid_alphas.append(alpha_val)
            
    # Return the lowest power threshold where instability begins
    return min(valid_alphas) if valid_alphas else np.nan

# Vectorize the engine to handle grids efficiently
vec_get_alpha = np.vectorize(get_critical_alpha)

# -------------------------------------------------------------------------
# 2. Setup Figure and Maximized 3D Layout
# -------------------------------------------------------------------------
fig = plt.figure(figsize=(11, 8.5))
fig.suptitle("Optomechanical Hopf Bifurcation Surface", fontsize=15, fontweight='bold')

# Create a single 3D plot spanning the figure
ax = fig.add_subplot(111, projection='3d')

# Push the plot upwards to leave a clean, wide workspace for sliders at the bottom
plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.25)

# -------------------------------------------------------------------------
# 3. Initial Parameter State & High-Resolution Grid
# -------------------------------------------------------------------------
init_delta = 1.0
init_nu = 1.0
init_gamma = 0.1

# 50x50 resolution grid for detailed plotting
delta_grid, nu_grid = np.meshgrid(np.linspace(-1.0, 3.0, 200), np.linspace(0.2, 8, 200))

# -------------------------------------------------------------------------
# 4. Rendering Function
# -------------------------------------------------------------------------
def render_plot(d_curr, n_curr, g_curr):
    ax.clear()
    
    # Calculate the threshold surface topology
    alpha_surface = vec_get_alpha(delta_grid, nu_grid, g_curr)
    
    # Mask out NaNs so the 3D surface engine renders seamlessly
    masked_surface = np.ma.masked_where(np.isnan(alpha_surface), alpha_surface)
    
    # Plot the main landscape
    surf = ax.plot_surface(delta_grid, nu_grid, masked_surface, cmap='plasma', 
                            edgecolor='none', alpha=0.85)
    
    # Track and plot the user's current specific coordinate state as a red sphere
    alpha_point = get_critical_alpha(d_curr, n_curr, g_curr)
    if not np.isnan(alpha_point):
        # FIXED: Removed 'weight' and replaced with 'markeredgewidth'
        ax.plot([d_curr], [n_curr], [alpha_point], 'ro', markersize=10, 
                markeredgecolor='k', markeredgewidth=2, zorder=100, label='Current Configuration')
        ax.legend(loc="upper left")
        
    # Labels and Boundaries
    ax.set_title(r"Lasing Threshold $\alpha_c(\delta, \nu)$", fontsize=12, pad=10)
    ax.set_xlabel(r"Detuning $\delta$", fontsize=10)
    ax.set_ylabel(r"Cavity Decay $\nu$", fontsize=10)
    ax.set_zlabel(r"Laser Power $\alpha_c$", fontsize=10)
    
    # Maintain a clear perspective view angle
    ax.view_init(elev=22, azim=-135)

# Initial draw
render_plot(init_delta, init_nu, init_gamma)

# -------------------------------------------------------------------------
# 5. Centered Interactive Sliders Panel
# -------------------------------------------------------------------------
ax_slider_delta = plt.axes([0.25, 0.14, 0.5, 0.025])
ax_slider_nu    = plt.axes([0.25, 0.09, 0.5, 0.025])
ax_slider_gamma = plt.axes([0.25, 0.04, 0.5, 0.025])

slider_delta = Slider(ax_slider_delta, r'Detuning $\delta$', -1.0, 3.0, valinit=init_delta, valfmt='%1.2f')
slider_nu    = Slider(ax_slider_nu, r'Cavity Decay $\nu$', 0.2, 2.5, valinit=init_nu, valfmt='%1.2f')
slider_gamma = Slider(ax_slider_gamma, r'Mech Damping $\gamma$', 0.01, 0.8, valinit=init_gamma, valfmt='%1.3f')

# Callback routine to refresh the visual asset dynamically
def update(val):
    d = slider_delta.val
    n = slider_nu.val
    g = slider_gamma.val
    render_plot(d, n, g)
    fig.canvas.draw_idle()

slider_delta.on_changed(update)
slider_nu.on_changed(update)
slider_gamma.on_changed(update)

plt.show()