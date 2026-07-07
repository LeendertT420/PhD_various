import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider
import umap
from equations import *
from classification import *


# =====================================================================
# CONFIGURATION: MATCH THIS SELECTION TO YOUR __MAIN__ SWEEP ORDERING
# =====================================================================
PARAM_ORDER = ["alpha", "delta", "sigma"]

umap_coords_grid = None


N = 15
sigma_val = 60

raw_filename = f'C:\\Users\\lion_remote\\Documents\\Geert\\results\\sweep_results_N={N}_sigma={sigma_val}.npz'
metrics_filename = f'C:\\Users\\lion_remote\\Documents\\Geert\\results\\sweep_results_N={N}_sigma={sigma_val}_metrics.npz'

# =====================================================================
# DATA DISCOVERY PIPELINE
# =====================================================================
print(f'loading data from {raw_filename}...')
data = np.load(raw_filename)
print('data succesfully loaded')


param_vectors = {
    "alpha": data['alphas'],
    "delta": data['deltas'],
}
states = data['states']          # Shape: (len_p1, len_p2, len_p3, 2N+1, num_steps)
above_thresh = data['above_threshold']
exploded = data['exploded']

if 'sigmas' in data and data['sigmas'].ndim > 0:
    param_vectors["sigma"] = data['sigmas']
    sigma_slider = True
else:
    # If the file saved it as a scalar or left it out, create a fallback 1-element array
    scalar_sigma = data['sigma'] if 'sigma' in data else 20  # replace 0.0 with your default
    param_vectors["sigma"] = np.array([scalar_sigma])
    sigma_slider = False

    # 2. Inject the singleton dimension at Axis 2 (the 'sigma' axis position)
    states = np.expand_dims(states, axis=2)              # Changes from (A, D, Var, Step) -> (A, D, 1, Var, Step)
    above_thresh = np.expand_dims(above_thresh, axis=2)  # Changes from (A, D) -> (A, D, 1)
    exploded = np.expand_dims(exploded, axis=2)

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
print(len(t_real))
dt = np.mean(np.diff(t_real))

freq_spectrum = np.sqrt(mu_spectrum(N)) / (2*np.pi)

# =====================================================================
# Computing metrics
# =====================================================================
metrics_data = np.load(metrics_filename, allow_pickle=True)
    
entropy_grid = metrics_data['entropy_grid']
N_peaks_grid = metrics_data['N_peaks_grid']
peak_positions = metrics_data['peak_positions'].item()
autocorr_grid = metrics_data['autocorr_grid']
classification_grid = metrics_data['classification_grid']
umap_coords_grid = metrics_data['umap_coords_grid']



# =====================================================================
# CANVAS AND 2X3 GRID ARCHITECTURE
# =====================================================================
fig, axs = plt.subplots(2, 4, figsize=(14, 9))
plt.subplots_adjust(bottom=0.25, hspace=0.35, wspace=0.25)

ax_time = axs[0, 0]
ax_Npeaks = axs[0, 1]
ax_complex = axs[1, 0]
ax_fft = axs[1, 1]
ax_entropy = axs[0, 2]
ax_autocor = axs[1, 2]
ax_class = axs[0,3]
ax_cluster = axs[1, 3]


init_indices = {name: len(vec) // 2 for name, vec in param_vectors.items()}

def extract_hyper_slice(tensor, indices_dict):
    selector = [indices_dict[name] for name in PARAM_ORDER]
    selector.append(slice(None))
    if len(tensor.shape) == 5: # For 5D states tensor
        selector.append(slice(None))
    return tensor[tuple(selector)]

current_slice = extract_hyper_slice(states, init_indices)

# --- PANEL 1: TIME SERIES INITIALIZATION ---
lines = [ax_time.plot(t_real, current_slice[i, :], label=f'$x_{i+1}$', alpha=0.3)[0] for i in range(N)]
z_line, = ax_time.plot(t_real, current_slice[2*N, :], 'k--', linewidth=1.5, label='Global $z$')
X_line, = ax_time.plot(t_real, np.sum(current_slice[:N,:], axis=0), 'k', linewidth=1.5, label='Global $X$')
ax_time.set_xlabel(r'Time ($t$)')
ax_time.set_ylabel('Amplitude')
ax_time.grid(True)
ax_time.legend(loc='upper right', fontsize='small')
ax_time.set_xlim(t_real[0], t_real[250])

# --- Initialize (delta, alpha) plane ---
d_grid = param_vectors["delta"]
a_grid = param_vectors["alpha"]

D_mesh, A_mesh = np.meshgrid(np.linspace(d_grid[0], d_grid[-1], len(d_grid)), np.linspace(a_grid[0], a_grid[-1], len(a_grid)))

# --- PANEL 2: Npeaks ---
mesh_Npeaks = ax_Npeaks.pcolormesh(D_mesh, A_mesh, np.zeros_like(D_mesh), shading='auto', cmap='viridis')
cbar_Npeaks = plt.colorbar(mesh_Npeaks, ax=ax_Npeaks)
cbar_Npeaks.set_label(r'Number of peaks in spectrum')

target_dot_Npeaks, = ax_Npeaks.plot(param_vectors["delta"][init_indices["delta"]], param_vectors["alpha"][init_indices["alpha"]], 'ro', markersize=8, label='Current State')
bif_upper_Npeaks, = ax_Npeaks.plot(param_vectors["delta"], upper_boundary(N, param_vectors["delta"]), 'k', linewidth=1.5, label='Bifurcation boundary')
bif_lower_Npeaks, = ax_Npeaks.plot(param_vectors["delta"], lower_boundary(N, param_vectors["delta"]), 'k', linewidth=1.5)
lasing_lines_Npeaks = [ax_Npeaks.plot(param_vectors["delta"], np.zeros_like(param_vectors["delta"]), 'r', linewidth=1)[0] for i in range(N)]
ax_Npeaks.set_xlabel(r'Detuning ($\delta$)')
ax_Npeaks.set_ylabel(r'Pump ($\alpha$)')
ax_Npeaks.set_title(r'Number of peaks in FFT spectrum')
ax_Npeaks.legend(bbox_to_anchor=(1, 1), fontsize='small')
ax_Npeaks.set_xlim(param_vectors["delta"][0], param_vectors["delta"][-1])
ax_Npeaks.set_ylim(param_vectors["alpha"][0], param_vectors["alpha"][-1])
ax_Npeaks.grid(False)

# --- PANEL 3: COMPLEX EIGENVALUE PLANE ---
ax_complex.axvline(0, color='r', linestyle='-', linewidth=1.5, alpha=0.7) # Stability Boundary Line
ax_complex.axhline(0, color='k', linestyle=':', alpha=0.5)
eig_scatter = ax_complex.scatter([], [], c='blue', edgecolors='k', s=45, zorder=3, label=r'$\lambda_i$')
ax_complex.set_xlabel(r'Real Part $\text{Re}(\lambda)$ (Damping)')
ax_complex.set_ylabel(r'Imaginary Part $\text{Im}(\lambda)$ (Frequency)')
ax_complex.set_title('Jacobian Eigenvalue Spectrum')
ax_complex.grid(True)


# --- PANEL 4: SPECTRAL FOURIER POWER SPECTRUM ---
fft_line, = ax_fft.plot([], [], color='purple', linewidth=1.5, zorder=10)
peaks, = ax_fft.plot([], [], 'ro', markersize=8)
spectrum = ax_fft.vlines(freq_spectrum, ymin=0, ymax=1, color='grey', alpha=0.5, linewidth=0.5)
ax_fft.set_xlabel(r'Frequency ($\omega$)')
ax_fft.set_ylabel('Power Spectral Density')
ax_fft.set_title(r'FFT Spectrum of Total Displacement $\sum_i x_i$')
ax_fft.set_xlim(0, 2*freq_spectrum[-1] -freq_spectrum[-2])
ax_fft.grid(False)


# --- PANEL 5: entropy ---
mesh_entropy = ax_entropy.pcolormesh(D_mesh, A_mesh, np.zeros_like(D_mesh), shading='auto', cmap='viridis')
cbar_entropy = plt.colorbar(mesh_entropy, ax=ax_entropy)
cbar_entropy.set_label(r'Shannon entropy')

target_dot_entropy, = ax_entropy.plot(param_vectors["delta"][init_indices["delta"]], param_vectors["alpha"][init_indices["alpha"]], 'ro', markersize=8, label='Current State')
bif_upper_entropy, = ax_entropy.plot(param_vectors["delta"], upper_boundary(N, param_vectors["delta"]), 'k', linewidth=1.5, label='Bifurcation boundary')
bif_lower_entropy, = ax_entropy.plot(param_vectors["delta"], lower_boundary(N, param_vectors["delta"]), 'k', linewidth=1.5)
lasing_lines_entropy = [ax_entropy.plot(param_vectors["delta"], np.zeros_like(param_vectors["delta"]), 'r', linewidth=1)[0] for i in range(N)]
ax_entropy.set_xlabel(r'Detuning ($\delta$)')
ax_entropy.set_ylabel(r'Pump ($\alpha$)')
ax_entropy.set_title(r'Entropy of FFT')
ax_entropy.set_xlim(param_vectors["delta"][0], param_vectors["delta"][-1])
ax_entropy.set_ylim(param_vectors["alpha"][0], param_vectors["alpha"][-1])
ax_entropy.grid(False)

# --- PANEL 6: entropy ---
mesh_autocor = ax_autocor.pcolormesh(D_mesh, A_mesh, np.zeros_like(D_mesh), shading='auto', cmap='viridis')
cbar_autocor = plt.colorbar(mesh_autocor, ax=ax_autocor)
cbar_autocor.set_label(r'Autocorrelation')

target_dot_autocor, = ax_autocor.plot(param_vectors["delta"][init_indices["delta"]], param_vectors["alpha"][init_indices["alpha"]], 'ro', markersize=8, label='Current State')
bif_upper_autocor, = ax_autocor.plot(param_vectors["delta"], upper_boundary(N, param_vectors["delta"]), 'k', linewidth=1.5, label='Bifurcation boundary')
bif_lower_autocor, = ax_autocor.plot(param_vectors["delta"], lower_boundary(N, param_vectors["delta"]), 'k', linewidth=1.5)
lasing_lines_autocor = [ax_autocor.plot(param_vectors["delta"], np.zeros_like(param_vectors["delta"]), 'r', linewidth=1)[0] for i in range(N)]
ax_autocor.set_xlabel(r'Detuning ($\delta$)')
ax_autocor.set_ylabel(r'Pump ($\alpha$)')
ax_autocor.set_title(r'Autocorrelation of time series')
ax_autocor.set_xlim(param_vectors["delta"][0], param_vectors["delta"][-1])
ax_autocor.set_ylim(param_vectors["alpha"][0], param_vectors["alpha"][-1])
ax_autocor.grid(False)


# --- PANEL 6: class ---
classification_colors = [
    '#2B2B2B',  # 0: Below Threshold (Muted Charcoal)
    '#1F77B4',  # 1: Single-Mode Lasing (Deep Blue)
    '#2CA02C',  # 2: Mode-Locked State (Vibrant Coherent Green)
    '#FF7F0E',  # 3: Multi-Mode Lasing (Warm Operational Orange)
    '#BCBCBC'   # 4: Chaotic State (Crimson Red)
]

state_labels = [
    "Below Threshold", 
    "Single-Mode", 
    "Mode-Locked", 
    "Multi-Mode", 
    "Chaos/unlabelled"
]

cmap_class = mcolors.ListedColormap(classification_colors)
bounds = [0, 1, 2, 3, 4, 5]
norm_class = mcolors.BoundaryNorm(bounds, cmap_class.N)

mesh_class = ax_class.pcolormesh(D_mesh, A_mesh, np.zeros_like(D_mesh), cmap=cmap_class, norm=norm_class, shading='nearest')
cbar_class = plt.colorbar(mesh_class, ax=ax_class, ticks=[0.5, 1.5, 2.5, 3.5, 4.5], spacing='proportional')
cbar_class.ax.set_yticklabels(state_labels)
cbar_class.ax.tick_params(length=0)

target_dot_class, = ax_class.plot(param_vectors["delta"][init_indices["delta"]], param_vectors["alpha"][init_indices["alpha"]], 'ro', markersize=8, label='Current State')
bif_upper_class, = ax_class.plot(param_vectors["delta"], upper_boundary(N, param_vectors["delta"]), 'k', linewidth=1.5, label='Bifurcation boundary')
bif_lower_class, = ax_class.plot(param_vectors["delta"], lower_boundary(N, param_vectors["delta"]), 'k', linewidth=1.5)
lasing_lines_class = [ax_class.plot(param_vectors["delta"], np.zeros_like(param_vectors["delta"]), 'r', linewidth=1)[0] for i in range(N)]
ax_class.set_xlabel(r'Detuning ($\delta$)')
ax_class.set_ylabel(r'Pump ($\alpha$)')
ax_class.set_title(r'Classification')
ax_class.set_xlim(param_vectors["delta"][0], param_vectors["delta"][-1])
ax_class.set_ylim(param_vectors["alpha"][0], param_vectors["alpha"][-1])
ax_class.grid(False)

# --- PANEL 8: UMAP SPECTRAL EMBEDDING SPACE ---
# Initialize an empty scatter layout. We will dynamically re-draw this on file swaps.
valid_mask = ~np.isnan(umap_coords_grid[:, :, 0, 0])
        
x_pts = umap_coords_grid[:, :, 0, 0][valid_mask]
y_pts = umap_coords_grid[:, :, 0, 1][valid_mask]
colors = classification_grid[:, :, 0][valid_mask]
        
if len(x_pts) > 0:
            # Render the entire data cloud statically
    ax_cluster.scatter(
    x_pts, y_pts, 
    c=colors, cmap=cmap_class, norm=norm_class, 
    edgecolors='none', s=25, alpha=0.4, zorder=2
    )
            # Re-scale graph frame padding tightly around the dataset coordinates
    ax_cluster.set_xlim(np.min(x_pts) - 0.8, np.max(x_pts) + 0.8)
    ax_cluster.set_ylim(np.min(y_pts) - 0.8, np.max(y_pts) + 0.8)

target_dot_cluster, = ax_cluster.plot([], [], 'ro', markersize=10, mec='k', mew=1.5, zorder=5, label='Current Selector')
ax_cluster.set_xlabel('UMAP Axis 1')
ax_cluster.set_ylabel('UMAP Axis 2')
ax_cluster.set_title('UMAP Latent Space Embedding')
ax_cluster.grid(True, linestyle=':', alpha=0.5)

# =====================================================================
# REAL-TIME SLIDER CONTROLS GENERATION
# =====================================================================
sliders = {}
slider_names = []
for idx, name in enumerate(['sigma', 'alpha', 'delta']):
    if not name == 'sigma' or sigma_slider == True:
        vec = param_vectors[name]
        ax_pos = plt.axes([0.15, 0.16 - (idx * 0.045), 0.7, 0.025])
        latex_label = rf'$\alpha$' if name == 'alpha' else (rf'$\delta$' if name == 'delta' else rf'$\sigma$')
        sliders[name] = Slider(ax=ax_pos, label=latex_label, valmin=vec[0], valmax=vec[-1], valinit=vec[init_indices[name]], valstep=vec)
        slider_names.append(name)

# =====================================================================
# INTERACTIVE RE-RENDERING ENGINE
# =====================================================================
def update_canvas(val):
    current_indices = {}
    current_vals = {}

    for name, vec in param_vectors.items():
        if name in slider_names:    
            current_vals[name] = sliders[name].val
            current_indices[name] = np.where(vec == sliders[name].val)[0][0]
        else:
            current_vals[name] = sigma_val
            current_indices[name] = 0
        
    # Slices selection allocation mapping
    c_slice = extract_hyper_slice(states, current_indices)
    
    # Extract structural flags mapping out of coordinates matrices indices
    a_i, s_i, d_i = current_indices['alpha'], current_indices['sigma'], current_indices['delta']
    
    # Dynamic key routing depending on how your main loop structured indices assignments
    # Re-map position based on PARAM_ORDER array indices routing maps
    map_idx = tuple([current_indices[n] for n in PARAM_ORDER])
    
    is_above = above_thresh[map_idx]
    is_exploded = exploded[map_idx]
    
    params = {'N': N,
              'gamma': np.ones(N)*0.05,
              'mu': mu_spectrum(N),
              'tau': 1,
              'xi': np.ones(N)}
    
    las_thres = lasing_threshold(params, param_vectors["delta"], return_all=False)


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
        fig.suptitle(rf"Superfluid Oscillator Array Control Panel ($\sigma$ = {current_vals['sigma']:.1f})", color='black', fontsize=14, weight='bold')
        
        # 1. Update Trajectory Lines
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
        ax_fft.vlines(freq_spectrum, ymin=0, ymax=np.max(fft_vals), color='grey', alpha=0.5)
        ax_fft.set_ylim(0, np.max(fft_vals))

        peaks_x, peaks_y = peak_positions[a_i, d_i, s_i]
        peaks.set_data(peaks_x, peaks_y)
        
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

    current_umap = umap_coords_grid[a_i, d_i, s_i] # Reads (a_i, d_i, 0)

    target_dot_cluster.set_data([current_umap[0]], [current_umap[1]])
    target_dot_cluster.set_visible(True)

        
    fig.canvas.draw_idle()

for name in slider_names:
    sliders[name].on_changed(update_canvas)

update_canvas(None)
plt.show()