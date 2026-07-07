import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.integrate import solve_ivp

# --- 1. Define ODE System ---
def ode_system(t, state, gamma, nu, delta, alpha):
    x, y, u, v = state
    # Ensure radical doesn't go negative during integration transient
    sqrt_alpha = np.sqrt(max(0.0, alpha))
    
    dxdt = y
    dydt = -gamma * y - x + u**2 + v**2
    dudt = (1.0 / nu) * (-u - (delta + x) * v + sqrt_alpha)
    dvdt = (1.0 / nu) * (-v + (delta + x) * u)
    return [dxdt, dydt, dudt, dvdt]

# --- 2. Solve Quartic Analytic Thresholds ---
def get_lasing_threshold(gamma_val, nu_val, delta_range):
    # Calculate polynomial coefficients for delta_eff
    a2 = gamma_val**2 * nu_val**2 - gamma_val * nu_val**3 + 2 * gamma_val * nu_val - 6 * nu_val**2 + 2 - (4 * nu_val / gamma_val)
    a1 = (1.0 / gamma_val) * (gamma_val * nu_val + 2)**2
    a0 = (nu_val**2 + gamma_val * nu_val + 1)**2
    
    alpha_thresholds = []
    delta_out = []
    
    for d in delta_range:
        # Coefficients of: delta_eff^4 + a2*delta_eff^2 + a1*d*delta_eff + a0 = 0
        coeffs = [1.0, 0.0, a2, a1 * d, a0]
        roots = np.roots(coeffs)
        # We only care about real solutions for delta_eff
        real_roots = roots[np.isreal(roots)].real
        
        for r in real_roots:
            alpha_c = (r - d) * (r**2 + 1.0)
            if alpha_c >= 0:  # Physical lasing threshold must be positive
                delta_out.append(d)
                alpha_thresholds.append(alpha_c)
                
    return np.array(delta_out), np.array(alpha_thresholds)

# --- 3. Generate Pre-computed 3D Discriminant Sheet ---
# We make a static surface grid to avoid slow 3D recalculations during slider moves
g_surf = np.logspace(-1, 0.7, 40)
n_surf = np.linspace(0.01, 2.5, 40)
LG_surf, N_surf = np.meshgrid(np.log10(g_surf), n_surf)
G_s = 10**LG_surf

GN_factor = G_s * N_surf
A_s = 27 * (GN_factor + 2)**8 / G_s**4 
a0_s = (N_surf**2 + GN_factor + 1)**2
a2_s = G_s**2 * N_surf**2 - G_s * N_surf**3 + 2 * G_s * N_surf - 6 * N_surf**2 + 2 - (4 * N_surf / G_s)
B_s = 4 * a2_s * ((G_s * N_surf + 2)**4 / G_s**2) * (a2_s**2 - 36 * a0_s)
C_s = -16 * a0_s * (a2_s**2 - 4 * a0_s)**2
disc_s = B_s**2 - 4 * A_s * C_s
r1 = (-B_s + np.sqrt(np.maximum(0, disc_s))) / (2 * A_s)
delta_sq_s = np.where(r1 >= 0, r1, np.nan)
Delta_pos_surf = np.sqrt(delta_sq_s)

# --- 4. Interactive Layout Setup ---
fig = plt.figure(figsize=(14, 9))
plt.subplots_adjust(bottom=0.3, hspace=0.4, wspace=0.3)

# Subplots
ax_time = fig.add_subplot(2, 2, 1)
ax_3d = fig.add_subplot(2, 2, 2, projection='3d')
ax_thresh = fig.add_subplot(2, 1, 2)

# Initial Slider Values
init_gamma = 1.2
init_nu = 0.5
init_delta = 0.4
init_alpha = 2.0

# Time properties
t_span = (0, 50)
t_eval = np.linspace(0, 50, 1000)
initial_state = [0.1, 0.0, 0.1, 0.0] # [x, y, u, v]

# --- Initial Plotting Call ---
# 1. Time plot
sol = solve_ivp(ode_system, t_span, initial_state, args=(init_gamma, init_nu, init_delta, init_alpha), t_eval=t_eval)
line_x, = ax_time.plot(sol.t, sol.y[0], color='dodgerblue', lw=2, label='$x(t)$')
ax_time.set_xlabel('Time ($t$)')
ax_time.set_ylabel('$x$')
ax_time.set_title('Time Evolution of Variable $x$')
ax_time.grid(True, linestyle='--', alpha=0.6)

# 2. 3D Parameter space plot
ax_3d.plot_surface(LG_surf, N_surf, Delta_pos_surf, cmap='viridis', alpha=0.3, rstride=2, cstride=2, antialiased=True)
ax_3d.plot_surface(LG_surf, N_surf, -Delta_pos_surf, cmap='viridis', alpha=0.3, rstride=2, cstride=2, antialiased=True)
point_3d, = ax_3d.plot([np.log10(init_gamma)], [init_nu], [init_delta], marker='o', color='crimson', markersize=8, label='Current State')
ax_3d.set_xlabel('$\log_{10}(\gamma)$')
ax_3d.set_ylabel(r'$\nu$')
ax_3d.set_zlabel(r'$\delta$')
ax_3d.set_title('3D Parameter Configuration Space')

# 3. Alpha-Delta threshold plot
d_range = np.linspace(-2.0, 2.0, 300)
d_vals, a_vals = get_lasing_threshold(init_gamma, init_nu, d_range)
scat_thresh = ax_thresh.scatter(d_vals, a_vals, color='purple', s=4, label=r'Threshold $\alpha_c$')
cross_point, = ax_thresh.plot([init_delta], [init_alpha], marker='X', color='crimson', markersize=10, label='Operating Point')
ax_thresh.set_xlabel(r'$\delta$')
ax_thresh.set_ylabel(r'$\alpha$')
ax_thresh.set_xlim(-2.0, 2.0)
ax_thresh.set_ylim(0, 10)
ax_thresh.set_title(r'Lasing Threshold Curve $\alpha_c(\delta)$')
ax_thresh.grid(True, linestyle='--', alpha=0.6)
ax_thresh.legend(loc='upper right')

# --- Sliders Configuration Layout ---
ax_color = 'lightgoldenrodyellow'
ax_gamma = plt.axes([0.15, 0.18, 0.3, 0.03], facecolor=ax_color)
ax_nu    = plt.axes([0.15, 0.13, 0.3, 0.03], facecolor=ax_color)
ax_delta = plt.axes([0.58, 0.18, 0.3, 0.03], facecolor=ax_color)
ax_alpha = plt.axes([0.58, 0.13, 0.3, 0.03], facecolor=ax_color)

s_gamma = Slider(ax_gamma, r'$\gamma$', 0.1, 4.0, valinit=init_gamma, valfmt='%1.2f')
s_nu    = Slider(ax_nu, r'$\nu$', 0.05, 2.0, valinit=init_nu, valfmt='%1.2f')
s_delta = Slider(ax_delta, r'$\delta$', -1.5, 1.5, valinit=init_delta, valfmt='%1.2f')
s_alpha = Slider(ax_alpha, r'$\alpha$', 0.0, 8.0, valinit=init_alpha, valfmt='%1.2f')

# --- Update Action on Slider Movement ---
def update(val):
    g = s_gamma.val
    n = s_nu.val
    d = s_delta.val
    a = s_alpha.val
    
    # Update ODE integration trace
    new_sol = solve_ivp(ode_system, t_span, initial_state, args=(g, n, d, a), t_eval=t_eval)
    line_x.set_ydata(new_sol.y[0])
    ax_time.set_ylim(min(new_sol.y[0]) - 0.2, max(new_sol.y[0]) + 0.2)
    
    # Update 3D tracking node coordinates
    point_3d.set_data_3d([np.log10(g)], [n], [d])
    
    # Update bottom 2D threshold plot markers and curves
    new_d, new_a = get_lasing_threshold(g, n, d_range)
    global scat_thresh
    scat_thresh.remove()
    scat_thresh = ax_thresh.scatter(new_d, new_a, color='purple', s=4)
    cross_point.set_data([d], [a])
    
    fig.canvas.draw_idle()

s_gamma.on_changed(update)
s_nu.on_changed(update)
s_delta.on_changed(update)
s_alpha.on_changed(update)

plt.show()