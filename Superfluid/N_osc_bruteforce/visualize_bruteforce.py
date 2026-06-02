import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from equations import mu_spectrum

# =====================================================================
# CONFIGURATION: MATCH THIS SELECTION TO YOUR __MAIN__ SWEEP ORDERING
# =====================================================================
# Simply change the string positions here if you reorder your sweep loops!
PARAM_ORDER = ["alpha", "delta", "sigma"]

# =====================================================================
# LOADING AND DATA DISCOVERY PIPELINE
# =====================================================================
data = np.load('sweep_results.npz')

# Extract parameter vectors from the archive
param_vectors = {
    "alpha": data['alphas'],
    "delta": data['deltas'],
    "sigma": data['sigmas']
}
states = data['states']  # Shape: (len_p1, len_p2, len_p3, 2N+1, num_time_steps)

# Discover physical parameters of the run
num_time_steps = states.shape[-1]
total_vars = states.shape[-2]
N = (total_vars - 1) // 2

# Map out which sweep axis position belongs to which parameter string
axis_mapping = {name: idx for idx, name in enumerate(PARAM_ORDER)}

# Re-compute the real-world timeline vector based on the saved shape
# (Matches your main script logic: t_span_eval timeline duration)
t_span = (0.0, 1000.0)
Dt_eval = 2*np.pi*5 / np.sqrt(mu_spectrum(N)[0]) # span ensures 5 oscillations of the heaviest (slowest) oscillator
t_span_eval = (t_span[1]-Dt_eval, t_span[1]) # evaluate on the last part of the simulation
t_res_eval = 2*np.pi / (np.sqrt(mu_spectrum(N)[-1]) * 20)
t_real = np.linspace(t_span_eval[0], t_span_eval[1], int(abs(t_span_eval[1] - t_span_eval[0])/t_res_eval))
print(t_real)

# =====================================================================
# CANVAS AND GRAPHICS ARCHITECTURE
# =====================================================================
fig, ax = plt.subplots(figsize=(11, 6))
plt.subplots_adjust(bottom=0.35)  # Room for stacked sliders below

# Initialize middle index selectors based on dynamic arrays
init_indices = {name: len(vec) // 2 for name, vec in param_vectors.items()}

def extract_hyper_slice(indices_dict):
    """Dynamically routes axis indices regardless of configuration order."""
    # Build an indexing tuple list: e.g., [init_indices['alpha'], init_indices['delta'], init_indices['sigma']]
    selector = [indices_dict[name] for name in PARAM_ORDER]
    selector.append(slice(None))  # Keep all state variables (2N + 1)
    selector.append(slice(None))  # Keep all time steps
    return states[tuple(selector)]

# Render initial state lines
initial_slice = extract_hyper_slice(init_indices)
lines = [ax.plot(t_real, initial_slice[i, :], label=f'$x_{i+1}$')[0] for i in range(N)]
z_line, = ax.plot(t_real, initial_slice[2*N, :], 'k--', linewidth=2, label='Global $z$')

ax.set_xlabel('Tail Time Evolution Steps')
ax.set_ylabel('Amplitude')
ax.legend(loc='upper right')
ax.grid(True)

# =====================================================================
# REAL-TIME INTERACTIVE SLIDER GENERATION
# =====================================================================
sliders = {}
slider_axes = []

# Dynamically space out sliders vertically below the viewport
for idx, name in enumerate(PARAM_ORDER):
    vec = param_vectors[name]
    # Calculate geometric screen anchors for the slider track row
    ax_pos = plt.axes([0.15, 0.22 - (idx * 0.05), 0.7, 0.03])
    slider_axes.append(ax_pos)
    
    # Beautiful LaTeX parameter rendering
    latex_label = rf'$\alpha$' if name == 'alpha' else (rf'$\delta$' if name == 'delta' else rf'$\sigma$')
    
    # Create the slider object bound to its specific matrix vector coordinates
    sliders[name] = Slider(
        ax=ax_pos,
        label=latex_label,
        valmin=vec[0],
        valmax=vec[-1],
        valinit=vec[init_indices[name]],
        valstep=vec
    )

# =====================================================================
# RE-RENDERING PIPELINE CALL INTERFACE
# =====================================================================
def update_canvas(val):
    # 1. Harvest current slider numerical positions and convert them to index numbers
    current_indices = {}
    current_vals = {}
    for name, vec in param_vectors.items():
        current_vals[name] = sliders[name].val
        current_indices[name] = np.where(vec == sliders[name].val)[0][0]
        
    # 2. Extract out the exact slice mapping
    current_slice = extract_hyper_slice(current_indices)
    
    # 3. Dynamic State Exception Checks (Threshold and Explosion Handling)
    if np.isnan(current_slice).all():
        # Everything is NaN: Means we tripped our custom above_threshold=False check
        ax.set_facecolor('#e6e6e6')  # Flat slate gray for unsimulated state space
        ax.set_title('💤 Region Omitted: Position Lies Below Lasing Threshold Bound')
    elif np.isnan(current_slice).any():
        # Fragmented NaNs: Means solve_ivp crashed mid-integration loop
        ax.set_facecolor('#ffcccc')  # Warning light red
        ax.set_title('⚠️ Simulation Exception: Parameter Coordinates Caused Chaotic Numerical Blow-up')
    else:
        # Standard Active Plot Execution
        ax.set_facecolor('#ffffff')  # Clean white
        ax.set_title(
            rf"Active Steady-State Orbit Tracking $\rightarrow$ "
            rf"$\alpha$: {current_vals['alpha']:.2f} | "
            rf"$\delta$: {current_vals['delta']:.2f} | "
            rf"$\sigma$: {current_vals['sigma']:.2f}"
        )
        
    # 4. Modify coordinate points arrays smoothly
    for i in range(N):
        lines[i].set_ydata(current_slice[i, :])
    z_line.set_ydata(current_slice[2*N, :])
    
    # 5. Dynamically fit graph frame to match wave height
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()

# Wire up updates to all generated slider components
for name in PARAM_ORDER:
    sliders[name].on_changed(update_canvas)

# Force immediate title execution check on load
update_canvas(None)
plt.show()