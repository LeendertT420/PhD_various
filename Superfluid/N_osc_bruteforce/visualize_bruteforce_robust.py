import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider
from equations import *
from classification import *

# =====================================================================
# CONFIGURATION & DISK IO ROUTING
# =====================================================================
PARAM_ORDER = ["alpha", "delta", "sigma"]

# Define all sigmas you have generated separate files for
AVAILABLE_SIGMAS = np.array([20, 40, 60, 80]) 

def get_filename(sigma_val):
    """Generates the target path dynamically based on slider values."""
    return f'./results/sweep_results_N=15_sigma={int(sigma_val)}.npz'

# =====================================================================
# GLOBAL STATE MEMORY SLOTS
# =====================================================================
states = None
above_thresh = None
exploded = None
eigvals_tensor = None
roots_tensor = None
has_stability_data = False

entropy_grid = None
N_peaks_grid = None
peak_positions = None
autocorr_grid = None
classification_grid = None

num_time_steps = 0
total_vars = 0
N = 0
t_real = None
dt = 0.0
freq_spectrum = None
current_loaded_sigma = None

# =====================================================================
# DYNAMIC SEPARATE-FILE LOADER ENGINE
# =====================================================================
def load_sigma_dataset(sigma_val):
    """Swaps data context in memory when the slider targets a new file."""
    global states, above_thresh, exploded, eigvals_tensor, roots_tensor, has_stability_data
    global entropy_grid, N_peaks_grid, peak_positions, autocorr_grid, classification_grid
    global num_time_steps, total_vars, N, t_real, dt, freq_spectrum, current_loaded_sigma
    
    filename = get_filename(sigma_val)
    print(f"🔄 Swapping memory contexts... Loading: {filename}")
    
    data = np.load(filename)
    
    # Isolate arrays
    states = data['states']
    above_thresh = data['above_threshold']
    exploded = data['exploded']
    
    # Append the dummy 3D axis at index 2 so existing tracking engines match shapes
    states = np.expand_dims(states, axis=2)
    above_thresh = np.expand_dims(above_thresh, axis=2)
    exploded = np.expand_dims(exploded, axis=2)
    
    has_stability_data = 'eigvals' in data
    if has_stability_data:
        eigvals_tensor = np.expand_dims(data['eigvals'], axis=2)
        roots_tensor = np.expand_dims(data['roots'], axis=2)
        
    num_time_steps = states.shape[-1]
    total_vars = states.shape[-2]
    N = (total_vars - 1) // 2
    
    t_real = data['time']
    dt = np.mean(np.diff(t_real))
    freq_spectrum = np.sqrt(mu_spectrum(N)) / (2 * np.pi)
    
    # Execute single-pass calculation grid update
    print("📊 Evaluating metrics for new file context...")
    entropy_grid = compute_grid_spectral_entropy(states, exploded, N)
    N_peaks_grid, peak_positions = compute_grid_peak_details(states, exploded, N, dt)
    autocorr_grid = compute_grid_autocorrelation_metrics(states, exploded, N, dt)
    classification_grid = compute_grid_classification(states, exploded, N, dt)
    
    current_loaded_sigma = sigma_val
    print("✅ Context loaded successfully.")

# Initialize the canvas context with the first available file
load_sigma_dataset(AVAILABLE_SIGMAS[0])

# Construct fixed global coordinate maps for sliders
param_vectors = {
    "alpha": data['alphas'] if 'data' in locals() else np.load(get_filename(AVAILABLE_SIGMAS[0]))['alphas'],
    "delta": data['deltas'] if 'data' in locals() else np.load(get_filename(AVAILABLE_SIGMAS[0]))['deltas'],
    "sigma": AVAILABLE_SIGMAS
}

# =====================================================================
# CANVAS AND 2X3 GRID ARCHITECTURE
# =====================================================================
fig, axs = plt.subplots(2, 4, figsize=(14, 9))
plt.subplots_adjust(bottom=0.25, hspace=0.35, wspace=0.25)

ax_time, ax_Npeaks, ax_entropy, ax_class = axs[0, 0], axs[0, 1], axs[0, 2], axs[0, 3]
ax_complex, ax_fft, ax_autocor = axs[1, 0], axs[1, 1], axs[1, 2]
axs[1, 3].axis('off') # Clear spare panel block

init_indices = {name: len(vec) // 2 for name, vec in param_vectors.items()}
init_indices["sigma"] = 0 # Match baseline array point indices

def extract_hyper_slice(tensor, indices_dict):
    selector = [indices_dict[name] for name in PARAM_ORDER]
    selector.append(slice(None))
    if len(tensor.shape) == 5:
        selector.append(slice(None))
    return tensor[tuple(selector)]

current_slice = extract_hyper_slice(states, init_indices)

# --- PANEL 1: TIME SERIES INITIALIZATION ---
lines = [ax_time.plot(t_real, current_slice[i, :], label=f'$x_{i+1}$', alpha=0.3)[0] for i in range(N)]
z_line, = ax_time.plot(t_real, current_slice[2*N, :], 'k--', linewidth=1.5, label='Global $z$')
X_line, = ax_time.plot(t_real, np.sum(current_slice[:N,:], axis=0), 'k', linewidth=1.5, label='Global $X$')
ax_time.set_xlabel('Time ($t$)')
ax_time.set_ylabel('Amplitude')
ax_time.grid(True)
ax_time.legend(loc='upper right', fontsize='small')
ax_time.set_xlim(t_real[0], t_real[250])

d_grid, a_grid = param_vectors["delta"], param_vectors["alpha"]
D_mesh, A_mesh = np.meshgrid(d_grid, a_grid)

# --- PANEL 2: Npeaks ---
mesh_Npeaks = ax_Npeaks.pcolormesh(D_mesh, A_mesh, np.zeros_like(D_mesh), shading='auto', cmap='viridis')
cbar_Npeaks = plt.colorbar(mesh_Npeaks, ax=ax_Npeaks)
target_dot_Npeaks, = ax_Npeaks.plot(d_grid[init_indices["delta"]], a_grid[init_indices["alpha"]], 'ro', markersize=8)
bif_upper_Npeaks, = ax_Npeaks.plot(d_grid, upper_boundary(N, d_grid), 'k', linewidth=1.5)
bif_lower_Npeaks, = ax_Npeaks.plot(d_grid, lower_boundary(N, d_grid), 'k', linewidth=1.5)
lasing_lines_Npeaks = [ax_Npeaks.plot(d_grid, np.zeros_like(d_grid), 'r', linewidth=1)[0] for _ in range(N)]
ax_Npeaks.set_xlim(d_grid[0], d_grid[-1])
ax_Npeaks.set_ylim(a_grid[0], a_grid[-1])

# --- PANEL 3: COMPLEX EIGENVALUE PLANE ---
ax_complex.axvline(0, color='r', linestyle='-', linewidth=1.5, alpha=0.7)
ax_complex.axhline(0, color='k', linestyle=':', alpha=0.5)
eig_scatter = ax_complex.scatter([], [], c='blue', edgecolors='k', s=45, zorder=3)

# --- PANEL 4: SPECTRAL FOURIER POWER SPECTRUM ---
fft_line, = ax_fft.plot([], [], color='purple', linewidth=1.5)
peaks, = ax_fft.plot([], [], 'ro', markersize=8)
ax_fft.grid(False)

# --- PANEL 5: Entropy ---
mesh_entropy = ax_entropy.pcolormesh(D_mesh, A_mesh, np.zeros_like(D_mesh), shading='auto', cmap='viridis')
cbar_entropy = plt.colorbar(mesh_entropy, ax=ax_entropy)
target_dot_entropy, = ax_entropy.plot(d_grid[init_indices["delta"]], a_grid[init_indices["alpha"]], 'ro', markersize=8)
bif_upper_entropy, = ax_entropy.plot(d_grid, upper_boundary(N, d_grid), 'k', linewidth=1.5)
bif_lower_entropy, = ax_entropy.plot(d_grid, lower_boundary(N, d_grid), 'k', linewidth=1.5)
lasing_lines_entropy = [ax_entropy.plot(d_grid, np.zeros_like(d_grid), 'r', linewidth=1)[0] for _ in range(N)]
ax_entropy.set_xlim(d_grid[0], d_grid[-1])
ax_entropy.set_ylim(a_grid[0], a_grid[-1])

# --- PANEL 6: Autocorrelation ---
mesh_autocor = ax_autocor.pcolormesh(D_mesh, A_mesh, np.zeros_like(D_mesh), shading='auto', cmap='viridis')
cbar_autocor = plt.colorbar(mesh_autocor, ax=ax_autocor)
target_dot_autocor, = ax_autocor.plot(d_grid[init_indices["delta"]], a_grid[init_indices["alpha"]], 'ro', markersize=8)
bif_upper_autocor, = ax_autocor.plot(d_grid, upper_boundary(N, d_grid), 'k', linewidth=1.5)
bif_lower_autocor, = ax_autocor.plot(d_grid, lower_boundary(N, d_grid), 'k', linewidth=1.5)
lasing_lines_autocor = [ax_autocor.plot(d_grid, np.zeros_like(d_grid), 'r', linewidth=1)[0] for _ in range(N)]
ax_autocor.set_xlim(d_grid[0], d_grid[-1])
ax_autocor.set_ylim(a_grid[0], a_grid[-1])

# --- PANEL 7: Classification ---
classification_colors = ['#2B2B2B', '#1F77B4', '#2CA02C', '#FF7F0E', '#D62728']
state_labels = ["Below Threshold (0)", "Single-Mode (1)", "Mode-Locked (2)", "Multi-Mode (3)", "Chaos (4)"]
cmap_class = mcolors.ListedColormap(classification_colors)
norm_class = mcolors.BoundaryNorm([0, 1, 2, 3, 4, 5], cmap_class.N)

mesh_class = ax_class.pcolormesh(D_mesh, A_mesh, np.zeros_like(D_mesh), cmap=cmap_class, norm=norm_class, shading='nearest')
cbar_class = plt.colorbar(mesh_class, ax=ax_class, ticks=[0.5, 1.5, 2.5, 3.5, 4.5], spacing='proportional')
cbar_class.ax.set_yticklabels(state_labels)
target_dot_class, = ax_class.plot(d_grid[init_indices["delta"]], a_grid[init_indices["alpha"]], 'ro', markersize=8)
bif_upper_class, = ax_class.plot(d_grid, upper_boundary(N, d_grid), 'k', linewidth=1.5)
bif_lower_class, = ax_class.plot(d_grid, lower_boundary(N, d_grid), 'k', linewidth=1.5)
lasing_lines_class = [ax_class.plot(d_grid, np.zeros_like(d_grid), 'r', linewidth=1)[0] for _ in range(N)]
ax_class.set_xlim(d_grid[0], d_grid[-1])
ax_class.set_ylim(a_grid[0], a_grid[-1])

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
    current_vals = {name: sliders[name].val for name in PARAM_ORDER}
    
    # 1. TRAP SIGMA SWAP EVAPORATION EVENT
    target_sigma = current_vals["sigma"]
    if target_sigma != current_loaded_sigma:
        load_sigma_dataset(target_sigma)
        
    # Get indices relative to global coordinate limits
    global_indices = {name: np.where(param_vectors[name] == current_vals[name])[0][0] for name in PARAM_ORDER}
    
    # CRITICAL MAPPER SHIFT: 
    # Because only ONE sigma is loaded in memory at a time, local matrix index is ALWAYS 0
    local_indices = global_indices.copy()
    local_indices['sigma'] = 0
    
    c_slice = extract_hyper_slice(states, local_indices)
    a_i, s_i, d_i = local_indices['alpha'], local_indices['sigma'], local_indices['delta'] # s_i is 0
    
    map_idx = tuple([local_indices[n] for n in PARAM_ORDER])
    
    is_above = above_thresh[map_idx]
    is_exploded = exploded[map_idx]
    
    params = {'N': N, 'gamma': np.ones(N)*0.05, 'mu': mu_spectrum(N), 'tau': 1, 'xi': np.ones(N)}
    las_thres = lasing_threshold(params, param_vectors["delta"], return_all=False)

    # Refresh structural mesh matrices configurations
    target_dot_Npeaks.set_data([current_vals["delta"]], [current_vals["alpha"]])
    lasing_lines_Npeaks[0].set_data(param_vectors["delta"], las_thres)
    N_peaks_slice = N_peaks_grid[:, :, s_i]
    mesh_Npeaks.set_array(N_peaks_slice.ravel(order='C'))
    mesh_Npeaks.set_clim(np.min(N_peaks_slice), np.max(N_peaks_slice))

    target_dot_entropy.set_data([current_vals["delta"]], [current_vals["alpha"]])
    lasing_lines_entropy[0].set_data(param_vectors["delta"], las_thres)
    entropy_slice = entropy_grid[:, :, s_i]
    mesh_entropy.set_array(entropy_slice.ravel(order='C'))
    mesh_entropy.set_clim(np.min(entropy_slice), np.max(entropy_slice))

    target_dot_autocor.set_data([current_vals["delta"]], [current_vals["alpha"]])
    lasing_lines_autocor[0].set_data(param_vectors["delta"], las_thres)
    autocorr_slice = autocorr_grid[:, :, s_i]
    mesh_autocor.set_array(autocorr_slice.ravel(order='C'))
    mesh_autocor.set_clim(np.min(autocorr_slice), np.max(autocorr_slice))

    target_dot_class.set_data([current_vals["delta"]], [current_vals["alpha"]])
    lasing_lines_class[0].set_data(param_vectors["delta"], las_thres)
    class_slice = classification_grid[:, :, s_i]
    mesh_class.set_array(class_slice.ravel(order='C'))

    if not is_above:
        fig.patch.set_facecolor('#f2f2f2')
        fig.suptitle('💤 Idle Matrix Regime: Target Coordinates Rest Below Lasing Threshold Bound', color='gray', fontsize=14, weight='bold')
        for k in range(N): lines[k].set_ydata(np.nan * t_real)
        z_line.set_ydata(np.nan * t_real)
        X_line.set_ydata(np.nan * t_real)
        eig_scatter.set_offsets(np.empty((0, 2)))
        fft_line.set_data([], [])
    elif is_exploded:
        fig.patch.set_facecolor('#ffe6e6')
        fig.suptitle('⚠️ Chaos Exception: Parameter Regime Triggered Explosive Instability Loop', color='red', fontsize=14, weight='bold')
        for k in range(N): lines[k].set_ydata(np.nan * t_real)
        z_line.set_ydata(np.nan * t_real)
        X_line.set_ydata(np.nan * t_real)
        eig_scatter.set_offsets(np.empty((0, 2)))
        fft_line.set_data([], [])
    else:
        fig.patch.set_facecolor('#ffffff')
        fig.suptitle(f"Superfluid Oscillator Array Control Panel ($\sigma$ = {current_vals['sigma']:.1f})", color='black', fontsize=14, weight='bold')
        
        for i in range(N):
            lines[i].set_ydata(c_slice[i, :])

        X = np.sum(c_slice[0:N, :], axis=0)
        z_line.set_ydata(c_slice[2*N, :])
        X_line.set_ydata(X)
        ax_time.relim()
        ax_time.autoscale_view()
        
        fft_vals = np.abs(np.fft.rfft((X-np.mean(X))/(np.max(X)-np.min(X))))
        freqs = np.fft.rfftfreq(num_time_steps, d=dt)
        
        fft_line.set_data(freqs, fft_vals)
        ax_fft.set_ylim(0, np.max(fft_vals))
        ax_fft.set_xlim(freqs[0], freqs[-1])

        # Lookup uses the global coordinate references tracking positions map
        peaks_x, peaks_y = peak_positions[global_indices['alpha'], global_indices['delta'], 0]
        peaks.set_data(peaks_x, peaks_y)
        
        if has_stability_data:
            current_eigs = extract_hyper_slice(eigvals_tensor, local_indices)
            flat_eigs = current_eigs.flatten()
            valid_eigs = flat_eigs[~np.isnan(flat_eigs)]
            
            if len(valid_eigs) > 0:
                scatter_points = np.column_stack((np.real(valid_eigs), np.imag(valid_eigs)))
                eig_scatter.set_offsets(scatter_points)
                ax_complex.set_xlim(np.min(np.real(valid_eigs)) - 0.2, np.max(np.real(valid_eigs)) + 0.2)
                ax_complex.set_ylim(np.min(np.imag(valid_eigs)) - 1.0, np.max(np.imag(valid_eigs)) + 1.0)
            else:
                eig_scatter.set_offsets(np.empty((0, 2)))
        
    fig.canvas.draw_idle()

for name in PARAM_ORDER:
    sliders[name].on_changed(update_canvas)

update_canvas(None)
plt.show()