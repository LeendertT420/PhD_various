import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp

# --- 1. Your Original Bifurcation Boundary Functions ---

def lower_boundary(N: int, d: float) -> float:
    if d**2 < 3:
        return 0.0
    s = np.sqrt(d**2 - 3)
    return -2/27 * (s - 2*d)**2 * (s + d) / N

def upper_boundary(N: int, d: float) -> float:
    if d**2 < 3:
        return 0.0
    s = np.sqrt(d**2 - 3)
    return 2/27 * (s + 2*d)**2 * (s - d) / N

# --- 2. Physics Backbone ---

def system_dynamics(t, state, alpha, delta, gamma, nu):
    x, y, u, v = state
    
    x_dot = y
    y_dot = -gamma * y - x + u**2 + v**2 
    
    u_dot = (1.0 / nu) * (-u - (delta + x) * v + np.sqrt(alpha))
    v_dot = (1.0 / nu) * (-v + (delta + x) * u)
    
    return [x_dot, y_dot, u_dot, v_dot]

def get_jacobian(state, delta, gamma, nu):
    x, y, u, v = state
    eff_detuning = delta + x
    
    J = np.zeros((4, 4))
    J[0, 1] = 1.0
    J[1, 0] = -1.0
    J[1, 1] = -gamma
    J[1, 2] = 2.0 * u
    J[1, 3] = 2.0 * v
    
    J[2, 0] = -v / nu
    J[2, 2] = -1.0 / nu
    J[2, 3] = -eff_detuning / nu
    
    J[3, 0] = u / nu
    J[3, 2] = eff_detuning / nu
    J[3, 3] = -1.0 / nu

    return J

def lasing_thresholds(deltas, gamma, nu):
    a_0 = (nu**2 + gamma*nu + 1)**2
    a_1 = nu*(gamma*nu+2)**2/gamma
    a_2 = gamma**2*nu**2 - gamma*nu**3 + 2*gamma*nu - 6*nu**2 + 2 - 4*nu/gamma

    all_roots = []
    for delta in deltas:
        roots = np.roots([1, 0, a_2, a_1*delta, a_0])
        all_roots.append(roots)
    
    all_roots = np.array(all_roots)
    mask = np.abs(np.imag(all_roots)) > 1e-8
    all_roots = np.where(mask, np.nan, np.real(all_roots))
    print(all_roots, np.shape(all_roots))

    thresholds = []
    for d_eff in all_roots.T:
        thresholds.append((d_eff-deltas)*(d_eff**2+1))

    return thresholds


def get_fixed_points(delta, alpha):
    roots = np.roots([1, 2*delta, delta**2+1, -alpha])
    return np.real(roots[np.isreal(roots)])

# --- 3. Interactive Plotter Setup ---
init_nu = 0.01
init_gamma = 0.05
init_delta = 0
init_alpha = 1

alpha_min, alpha_max = 0.01, 5
delta_min, delta_max = -5, 5

fig, axs = plt.subplots(2, 2, figsize=(11, 8))
plt.subplots_adjust(bottom=0.25, hspace=0.35, wspace=0.25)

ax_bif, ax_time = axs[0, 0], axs[0, 1]
ax_eig, ax_fft = axs[1, 0], axs[1, 1]

# Recompute Background Using Restored Functions
delta_grid = np.linspace(delta_min, delta_max, 500)
low_a = np.array([lower_boundary(1, d) for d in delta_grid])
upp_a = np.array([upper_boundary(1, d) for d in delta_grid])

# Panel 1 Layout
ax_bif.fill_between(delta_grid, low_a, upp_a, color='gray', alpha=0.3, label='Bifurcation')
current_pos_dot, = ax_bif.plot([init_delta], [init_alpha], 'ro', markersize=8, label='Current Point')
lasing_lines = [ax_bif.plot([], [], 'r', lw=1)[0] for _ in range(4)]
ax_bif.set_xlim(delta_min, delta_max)
ax_bif.set_ylim(alpha_min, alpha_max)
ax_bif.set_xlabel(r'Detuning $\delta$')
ax_bif.set_ylabel(r'Drive Power $\alpha$')
ax_bif.legend(loc='upper left')
ax_bif.grid(True)

# Panel 2 Layout
time_line, = ax_time.plot([], [], 'b-', lw=1.5)
ax_time.set_xlabel('Time $t$')
ax_time.set_ylabel('Displacement $x(t)$')
ax_time.grid(True)

# Panel 3 Layout
eig_scatter, = ax_eig.plot([], [], 'kx', markersize=8, mew=2)
ax_eig.axvline(0, color='r', linestyle='--', alpha=0.5)
ax_eig.set_xlabel(r'Real Part $\text{Re}(\lambda)$')
ax_eig.set_ylabel(r'Imag Part $\text{Im}(\lambda)$')
ax_eig.grid(True)

# Panel 4 Layout
fft_line, = ax_fft.plot([], [], 'm-', lw=1.5)
ax_fft.set_xlabel('Frequency $f$')
ax_fft.set_ylabel('Power Spectrum $|X(f)|^2$')
ax_fft.set_yscale('log')
ax_fft.grid(True)

# --- 4. Dynamic Updates ---

def update(val):
    nu = s_nu.val
    gamma = s_gamma.val
    delta = s_delta.val
    alpha = s_alpha.val
    x0 = s_init.val
    
    current_pos_dot.set_data([delta], [alpha])
    thresholds = lasing_thresholds(delta_grid, gamma, nu)
    for lasing_line, threshold in zip(lasing_lines, thresholds):
        lasing_line.set_data(delta_grid, threshold)
    
    t_span = (0, 600)
    t_eval = np.linspace(t_span[0], t_span[1], 2000)
    x_star = np.max(get_fixed_points(delta, alpha))
    u_star = x_star / np.sqrt(alpha)
    v_star = u_star * (delta + x_star)
    
    # 2. Add a tiny perturbation so it isn't perfectly static
    eps = 1e-4
    init_state = [0, 0, 0, 0]
    init_state = [x_star + eps, eps, u_star, v_star]
    init_state = [20, 20, 20, 20]
    init_state = [x0*x_star, 0, x0*u_star, x0*v_star]
    
    sol = solve_ivp(
        system_dynamics, t_span, init_state, t_eval=t_eval,
        args=(alpha, delta, gamma, nu), method='RK45'
    )
    
    x_trace = sol.y[0]
    t_trace = sol.t
    
    steady_slice = int(len(x_trace) * 0.3)
    x_steady = x_trace[steady_slice:]
    
    ax_time.set_xlim(t_span[0], t_span[1])
    if len(x_trace) > 0:
        ax_time.set_ylim(np.min(x_trace) - 0.2, np.max(x_trace) + 0.2)
    time_line.set_data(t_trace, x_trace)
    
    x_star = np.max(get_fixed_points(delta, alpha))
    J_matrix = get_jacobian([x_star, 0, x_star/np.sqrt(alpha), np.sqrt(x_star*(1-x_star/alpha))], delta, gamma, nu)
    eigenvalues = np.linalg.eigvals(J_matrix)
    
    eig_scatter.set_data(np.real(eigenvalues), np.imag(eigenvalues))
    ax_eig.set_xlim(- 0.2, 0.2)
    ax_eig.set_ylim(-2.5, 2.5)
    
    dt = t_eval[1] - t_eval[0]
    n = len(x_steady)
    fft_vals = np.fft.rfft(x_steady - np.mean(x_steady))
    frequencies = np.fft.rfftfreq(n, d=dt)
    power_spectrum = np.abs(fft_vals)**2
    
    fft_line.set_data(frequencies, power_spectrum)
    ax_fft.set_xlim(0, 0.5)
    if len(power_spectrum) > 0:
        ax_fft.set_ylim(max(1e-6, np.min(power_spectrum)), np.max(power_spectrum) * 5)
        
    fig.canvas.draw_idle()

# --- 5. Controls Placement ---

axcolor = 'lightgray'
ax_slider_nu = plt.axes([0.15, 0.14, 0.3, 0.03], facecolor=axcolor)
ax_slider_gamma = plt.axes([0.15, 0.09, 0.3, 0.03], facecolor=axcolor)
ax_slider_delta = plt.axes([0.6, 0.14, 0.3, 0.03], facecolor=axcolor)
ax_slider_alpha = plt.axes([0.6, 0.09, 0.3, 0.03], facecolor=axcolor)
ax_slider_init = plt.axes([0.6, 0.04, 0.3, 0.03], facecolor=axcolor)

s_nu = Slider(ax_slider_nu, r'Cavity Decay $\nu$', 0.001, 3.0, valinit=init_nu)
s_gamma = Slider(ax_slider_gamma, r'Mech Damping $\gamma$', 0.01, 1.0, valinit=init_gamma)
s_delta = Slider(ax_slider_delta, r'Detuning $\delta$', delta_min, delta_max, valinit=init_delta)
s_alpha = Slider(ax_slider_alpha, r'Laser Power $\alpha$', alpha_min, alpha_max, valinit=init_alpha)
s_init = Slider(ax_slider_init, r'Initial condition', 0, 20, valinit=0)

s_nu.on_changed(update)
s_gamma.on_changed(update)
s_delta.on_changed(update)
s_alpha.on_changed(update)
s_init.on_changed(update)

update(None)
plt.show()