import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from equations import *

# =====================================================================
# CONFIGURATION: MATCH THIS SELECTION TO YOUR __MAIN__ SWEEP ORDERING
# =====================================================================
PARAM_ORDER = ["alpha", "delta", "sigma"]

# =====================================================================
# DATA DISCOVERY PIPELINE
# =====================================================================
data = np.load('sweep_results_N=15.npz')
print('KANKER:', data.keys)

param_vectors = {
    "alpha": data['alphas'],
    "delta": data['deltas'],
    "sigma": data['sigmas']
}
states = data['states']          # Shape: (len_p1, len_p2, len_p3, 2N+1, num_steps)
above_thresh = data['above_threshold']
exploded = data['exploded']
print(exploded)


# Dense pad array collection checks for stability tracking
has_stability_data = 'eigvals' in data
if has_stability_data:
    eigvals_tensor = data['eigvals']  # Shape: (len_p1, len_p2, len_p3, max_roots, 2N+1)
    roots_tensor = data['roots']

num_time_steps = states.shape[-1]
total_vars = states.shape[-2]
N = (total_vars - 1) // 2

# Assume a default dt sampling rate based on your main script construction
t_real = data['time']
dt = np.mean(np.diff(t_real))

# =====================================================================
# CANVAS AND 2X2 GRID ARCHITECTURE
# =====================================================================
fig, axs = plt.subplots(2, 2, figsize=(14, 9))
plt.subplots_adjust(bottom=0.25, hspace=0.35, wspace=0.25)

ax_time = axs[0, 0]
ax_plane = axs[0, 1]
ax_complex = axs[1, 0]
ax_fft = axs[1, 1]

init_indices = {name: len(vec) // 2 for name, vec in param_vectors.items()}

def extract_hyper_slice(tensor, indices_dict):
    selector = [indices_dict[name] for name in PARAM_ORDER]
    selector.append(slice(None))
    if len(tensor.shape) == 5: # For 5D states tensor
        selector.append(slice(None))
    return tensor[tuple(selector)]

current_slice = extract_hyper_slice(states, init_indices)

# --- PANEL 1: TIME SERIES INITIALIZATION ---
lines = [ax_time.plot(t_real, current_slice[i, :], label=f'$x_{i+1}$')[0] for i in range(N)]
z_line, = ax_time.plot(t_real, current_slice[2*N, :], 'k--', linewidth=1.5, label='Global $z$')
ax_time.set_xlabel('Time ($t$)')
ax_time.set_ylabel('Amplitude')
ax_time.grid(True)
ax_time.legend(loc='upper right', fontsize='small')

# --- PANEL 2: (DELTA, ALPHA) PARAMETER PLANE ---
d_grid = param_vectors["delta"]
a_grid = param_vectors["alpha"]
D_mesh, A_mesh = np.meshgrid(np.linspace(d_grid[0], d_grid[-1], 200), np.linspace(a_grid[0], a_grid[-1], 200))


# Shade stable vs active region boundaries
#ax_plane.contourf(D_mesh, A_mesh, A_mesh > Threshold_mesh, levels=[-0.5, 0.5, 1.5], colors=['#e6e6e6', '#e6f2ff'], alpha=0.5)
#ax_plane.plot(np.linspace(d_grid[0], d_grid[-1], 100), z_crit * (np.linspace(d_grid[0], d_grid[-1], 100)**2 + 1.0), 'b--', label='Lasing Threshold')
# Dynamic location target marker
target_dot, = ax_plane.plot(param_vectors["delta"][init_indices["delta"]], param_vectors["alpha"][init_indices["alpha"]], 'ro', markersize=8, label='Current State')
ax_plane.set_xlabel(r'Detuning ($\delta$)')
ax_plane.set_ylabel(r'Pump ($\alpha$)')
ax_plane.set_title(r'Parameter Landscape $(\delta, \alpha)$')
ax_plane.legend(loc='lower right', fontsize='small')
ax_plane.grid(True)

# --- PANEL 3: COMPLEX EIGENVALUE PLANE ---
ax_complex.axvline(0, color='r', linestyle='-', linewidth=1.5, alpha=0.7) # Stability Boundary Line
ax_complex.axhline(0, color='k', linestyle=':', alpha=0.5)
eig_scatter = ax_complex.scatter([], [], c='blue', edgecolors='k', s=45, zorder=3, label=r'$\lambda_i$')
ax_complex.set_xlabel(r'Real Part $\text{Re}(\lambda)$ (Damping)')
ax_complex.set_ylabel(r'Imaginary Part $\text{Im}(\lambda)$ (Frequency)')
ax_complex.set_title('Jacobian Eigenvalue Spectrum')
ax_complex.grid(True)

# --- PANEL 4: SPECTRAL FOURIER POWER SPECTRUM ---
fft_line, = ax_fft.plot([], [], color='purple', linewidth=1.5)
ax_fft.set_xlabel('Frequency ($\omega$)')
ax_fft.set_ylabel('Power Spectral Density')
ax_fft.set_title(r'FFT Spectrum of Total Displacement $\sum x_i$')
ax_fft.grid(True)

# =====================================================================
# REAL-TIME SLIDER CONTROLS GENERATION
# =====================================================================
sliders = {}
for idx, name in enumerate(PARAM_ORDER):
    vec = param_vectors[name]
    ax_pos = plt.axes([0.15, 0.16 - (idx * 0.045), 0.7, 0.025])
    latex_label = rf'$\alpha$' if name == 'alpha' else (rf'$\delta$' if name == 'delta' else rf'$\sigma$')
    sliders[name] = Slider(ax=ax_pos, label=latex_label, valmin=vec[0], valmax=vec[-1], valinit=vec[init_indices[name]], valstep=vec)

# =====================================================================
# INTERACTIVE RE-RENDERING ENGINE
# =====================================================================
def update_canvas(val):
    current_indices = {}
    current_vals = {}
    for name, vec in param_vectors.items():
        current_vals[name] = sliders[name].val
        current_indices[name] = np.where(vec == sliders[name].val)[0][0]
        
    # Slices selection allocation mapping
    c_slice = extract_hyper_slice(states, current_indices)
    
    # Extract structural flags mapping out of coordinates matrices indices
    a_i, s_i, d_i = current_indices['alpha'], current_indices['sigma'], current_indices['delta']
    # Dynamic key routing depending on how your main loop structured indices assignments
    # Re-map position based on PARAM_ORDER array indices routing maps
    map_idx = tuple([current_indices[n] for n in PARAM_ORDER])
    
    is_above = above_thresh[map_idx]
    is_exploded = exploded[map_idx]
    
    # Move the parameter marker point 
    target_dot.set_data([current_vals["delta"]], [current_vals["alpha"]])
    
    # Handle global panel states rendering exceptions cleanly
    if not is_above:
        fig.patch.set_facecolor('#f2f2f2')
        fig.suptitle('💤 Idle Matrix Regime: Target Coordinates Rest Below Lasing Threshold Bound', color='gray', fontsize=14, weight='bold')
        for k in range(N): lines[k].set_ydata(np.nan * t_real)
        z_line.set_ydata(np.nan * t_real)
        eig_scatter.set_offsets(np.empty((0, 2)))
        fft_line.set_data([], [])
    elif is_exploded:
        fig.patch.set_facecolor('#ffe6e6')
        fig.suptitle('⚠️ Chaos Exception: Parameter Regime Triggered Explosive Instability Loop', color='red', fontsize=14, weight='bold')
        for k in range(N): lines[k].set_ydata(np.nan * t_real)
        z_line.set_ydata(np.nan * t_real)
        eig_scatter.set_offsets(np.empty((0, 2)))
        fft_line.set_data([], [])
    else:
        fig.patch.set_facecolor('#ffffff')
        fig.suptitle(f"Superfluid Oscillator Array Control Panel ($\sigma$ = {current_vals['sigma']:.1f})", color='black', fontsize=14, weight='bold')
        
        # 1. Update Trajectory Lines
        for i in range(N):
            lines[i].set_ydata(c_slice[i, :])
        z_line.set_ydata(c_slice[2*N, :])
        ax_time.relim()
        ax_time.autoscale_view()
        
        # 2. Compute and Update Fast Fourier Transform
        total_displacement = np.sum(c_slice[0:N, :], axis=0)
        # Apply Hanning window to prevent edge leak artifacts across small sample slices
        windowed_signal = total_displacement * np.hanning(num_time_steps)
        fft_vals = np.abs(np.fft.rfft(windowed_signal))
        freqs = np.fft.rfftfreq(num_time_steps, d=dt)
        
        fft_line.set_data(freqs, fft_vals)
        ax_fft.relim()
        ax_fft.autoscale_view()
        
        # 3. Update Eigenvalue Scatter Points (If tracked and logged)
        if has_stability_data:
            current_eigs = extract_hyper_slice(eigvals_tensor, current_indices) # Shape: (max_roots, 2N+1)
            # Flatten array and filter out empty NaN padding indicators
            flat_eigs = current_eigs.flatten()
            valid_eigs = flat_eigs[~np.isnan(flat_eigs)]
            
            if len(valid_eigs) > 0:
                scatter_points = np.column_stack((np.real(valid_eigs), np.imag(valid_eigs)))
                eig_scatter.set_offsets(scatter_points)
                
                # Dynamic scaling adjustment for stability spectrum window
                ax_complex.set_xlim(np.min(np.real(valid_eigs)) - 0.2, np.max(np.real(valid_eigs)) + 0.2)
                ax_complex.set_ylim(np.min(np.imag(valid_eigs)) - 1.0, np.max(np.imag(valid_eigs)) + 1.0)
            else:
                eig_scatter.set_offsets(np.empty((0, 2)))
        
    fig.canvas.draw_idle()

for name in PARAM_ORDER:
    sliders[name].on_changed(update_canvas)

update_canvas(None)
plt.show()