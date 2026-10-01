import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from matplotlib.gridspec import GridSpec
from scipy.signal import savgol_filter, hilbert
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

# --- Matplotlib Styling ---
plt.rcParams.update({'mathtext.fontset': 'cm'})
plt.rcParams.update({'font.family': 'STIXGeneral'})
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'axes.xmargin': 0})


def remove_slow_offset_auto(xs, dt, freqs, poly=3, period_factor=8):
    """
    Automatically choose Savitzky–Golay window per mode
    based on estimated oscillation frequency.
    """
    N_modes, N_t = xs.shape
    xs_detrended = np.zeros_like(xs)
    trends = np.zeros_like(xs)
    windows = []

    for i in range(N_modes):
        f = freqs[i]

        if f == 0:
            window = 101
        else:
            period_samples = int(1 / (f * dt))
            window = period_factor * period_samples

        window = max(window, poly + 2)
        if window % 2 == 0:
            window += 1
        window = min(window, N_t - (1 - N_t % 2))

        trend = savgol_filter(xs[i], window_length=window, polyorder=poly)

        xs_detrended[i] = xs[i] - trend
        trends[i] = trend
        windows.append(window)

    return xs_detrended, trends, np.array(windows), freqs


def get_phase(t, x):
    analytic = hilbert(x)
    return np.unwrap(np.angle(analytic))


def get_mode_locking_order_parameter(xs_array):
    """
    Computes an AMPLITUDE-WEIGHTED Mode-Locking Order Parameter R_ML(t) in [0, 1].
    This ignores phase noise from weak/zero-amplitude modes.
    xs_array: shape (N_modes, N_time_steps)
    """
    # Get complex analytic signals
    zs = hilbert(xs_array, axis=1)
    
    # Calculate z_{n+1} * conjugate(z_n)
    # The angle of this product is exactly (phi_{n+1} - phi_n)
    # The magnitude is A_{n+1} * A_n (this provides the natural amplitude weighting)
    adjacent_cross_terms = zs[1:] * np.conj(zs[:-1])
    
    # Coherent sum (magnitudes of the summed vectors)
    coherent_sum = np.abs(np.sum(adjacent_cross_terms, axis=0))
    
    # Incoherent sum (sum of the magnitudes)
    incoherent_sum = np.sum(np.abs(adjacent_cross_terms), axis=0)
    
    # Safe division
    R_ML_weighted = np.divide(
        coherent_sum, 
        incoherent_sum, 
        out=np.zeros_like(coherent_sum), 
        where=incoherent_sum!=0
    )
    
    return R_ML_weighted


def get_instantaneous_pulse_coherence(xs_array):
    """
    Computes Instantaneous Coherence R_pulse(t) in [0, 1].
    Peaks EXACTLY at 1 during pulse max, drops between pulses.
    xs_array: shape (N_modes, N_time_steps)
    """
    zs = hilbert(xs_array, axis=1)  # shape (N_modes, N_t)
    coherent_sum = np.abs(np.sum(zs, axis=0))
    incoherent_sum = np.sum(np.abs(zs), axis=0)
    R_pulse = np.divide(coherent_sum, incoherent_sum, out=np.zeros_like(coherent_sum), where=incoherent_sum!=0)
    return R_pulse


# --- Data Loading & Preprocessing ---
filename = "..\simulations\data\stable_mode_lock\simulation_one_mode_2026-07-30_14-02-42-665190.json"

with open(filename, "r") as f:
    data = json.load(f)

t = np.array(data["time"])[:]
I = np.array(data["intensity"])[:]
x = np.array(data["x"])[:]
omegas = np.array(data["parameters"]["omegas"])
mode_freqs = omegas / (2 * np.pi)

print(f"Found mode freqs: {mode_freqs}")

try:
    xs = np.array(data["xs"])[:, :]
    print(f"Traces shape: {xs.shape}")
except Exception:
    print("Individual traces not saved for this sim")

N = xs.shape[0]

t_pulse_start = 0
t_pulse_stop = 1.0

mask = (t > t_pulse_start) & (t < t_pulse_stop)

t = t[mask]
x = -x[mask]
xs = -xs[:, mask]

xs_detrended = xs

ns = np.arange(1, 12)

t_begin = 0.0
t_end = 1.0

mask_piece = (t > t_begin) & (t < t_end)
t_piece = t[mask_piece]

phases = []
for n in tqdm(ns, desc="Calculating phases"):
    x_piece = xs_detrended[n - 1, mask_piece]
    phi = get_phase(t_piece, x_piece)
    phases.append(phi)
phases = np.array(phases)

phases_corrected = []
plt.figure()
for n in tqdm(ns[:], desc="Correcting phases"):
    print(f"n = {n}, phases[]")
    f = phases[n - 1, :]
    f -= np.average(f[-100000:])
    phases_corrected.append(f)
    plt.plot(f)
plt.title("Phase Drift vs Time (Close window to launch animation)")
plt.show()

phases_corrected = np.array(phases_corrected)
T = np.shape(phases_corrected)[1]  # Number of time steps

# --- Compute Total Amplitude & Order Parameters ---
xs_piece = xs_detrended[:len(ns), mask_piece]
total_x = np.sum(xs_piece, axis=0)

# Pass xs_piece instead of phases to the new weighted function
R_ML = get_mode_locking_order_parameter(xs_piece)
# Compute a running average of R_ML over 5% of the data window
window_size = max(T // 100, 10) 
R_ML_running_avg = uniform_filter1d(R_ML, size=window_size)

R_pulse = get_instantaneous_pulse_coherence(xs_piece)

# --- Dual Plot Layout Setup with GridSpec ---
fig = plt.figure(figsize=(15, 8))
# 3 rows, 2 columns. Left column spans all 3 rows.
gs = GridSpec(3, 2, figure=fig, width_ratios=[1, 1.3], bottom=0.20, top=0.92, left=0.06, right=0.96, wspace=0.25, hspace=0.2)

ax_circle = fig.add_subplot(gs[:, 0])
ax_wave = fig.add_subplot(gs[0, 1])
ax_rml = fig.add_subplot(gs[1, 1], sharex=ax_wave)
ax_rpulse = fig.add_subplot(gs[2, 1], sharex=ax_wave)

# Helper function for scatter points
def get_points(idx):
    z = np.exp(1j * phases_corrected[:, idx])
    return np.real(z), np.imag(z)

mode_colors = plt.cm.turbo(np.linspace(0.05, 0.95, len(ns)))

# Initial points (frame 0)
xs_0, ys_0 = get_points(0)
x_avg_0, y_avg_0 = np.mean(xs_0), np.mean(ys_0)

# 1. Left Subplot: Unit Circle Phase Visualization
unit_circle_theta = np.linspace(0, 2 * np.pi, 200)
ax_circle.plot(np.cos(unit_circle_theta), np.sin(unit_circle_theta), 'k--', alpha=0.35, label='Unit Circle')

scat = ax_circle.scatter(
    xs_0, ys_0, c=mode_colors, s=75, zorder=3, edgecolors='black', 
    linewidths=0.6, label=f'Oscillators ({len(ns)} modes)'
)

avg_dot_circle, = ax_circle.plot(
    [x_avg_0], [y_avg_0], marker='D', color='crimson', ms=9, 
    markeredgecolor='black', markeredgewidth=1.2, zorder=4, 
    linestyle='None', label='2D Mean Position'
)

mean_line, = ax_circle.plot(
    [0, x_avg_0], [0, y_avg_0], color='crimson', linestyle=':', lw=1.5, alpha=0.7, zorder=3
)

ax_circle.set_xlim(-1.35, 1.35)
ax_circle.set_ylim(-1.35, 1.35)
ax_circle.set_aspect('equal')
ax_circle.set_title("Relative Phases & 2D Mean Vector")
ax_circle.set_xlabel(r"$\cos(\phi)$")
ax_circle.set_ylabel(r"$\sin(\phi)$")
ax_circle.grid(True, alpha=0.2)
ax_circle.legend(loc='upper left', framealpha=0.88, fontsize=10)

# 2. Right Subplots: Time Series
# Row 1: Total Amplitude

for i in range(1, xs.shape[0] + 1):
    ax_wave.plot(t_piece, xs[i - 1, :], lw = 1, alpha = 0.5, color = "tab:blue")
ax_wave.plot(t_piece, total_x, 'k-', lw=1, alpha=0.75)
dot_wave, = ax_wave.plot([t_piece[0]], [total_x[0]], 'ro', ms=7, zorder=4)
vline_wave = ax_wave.axvline(t_piece[0], color='red', linestyle='--', alpha=0.6)
ax_wave.set_xlim(t_piece[0], t_piece[-1])
ax_wave.set_ylim(np.min(total_x) * 1.15, np.max(total_x) * 1.15)
ax_wave.set_ylabel(r"Total Amp $\sum x_i$")
ax_wave.set_title("System Dynamics", fontweight='bold')
ax_wave.grid(True, alpha=0.3)
plt.setp(ax_wave.get_xticklabels(), visible=False)

# Row 2: Mode-Locking Order Parameter R_ML & Running Average
ax_rml.plot(t_piece, R_ML, 'tab:green', lw=1.5, alpha=0.35, label="Instantaneous")
ax_rml.plot(t_piece, R_ML_running_avg, 'darkgreen', lw=2.5, alpha=0.9, label="Running Average")
hline_thresh = ax_rml.axhline(0.8, color='orange', linestyle='--', alpha=0.8, label='ML Threshold') # Added Threshold Line
dot_rml, = ax_rml.plot([t_piece[0]], [R_ML[0]], 'ro', ms=5, alpha=0.5, zorder=4)
dot_rml_avg, = ax_rml.plot([t_piece[0]], [R_ML_running_avg[0]], 'D', color='darkgreen', ms=7, zorder=5)
vline_rml = ax_rml.axvline(t_piece[0], color='red', linestyle='--', alpha=0.6)
ax_rml.set_ylim(-0.05, 1.05)
ax_rml.set_ylabel(r"$R_{\mathrm{ML}}(t)$")
ax_rml.legend(loc='lower right', framealpha=0.88, fontsize=10)
ax_rml.grid(True, alpha=0.3)
plt.setp(ax_rml.get_xticklabels(), visible=False)

# Row 3: Instantaneous Pulse Coherence R_pulse
ax_rpulse.plot(t_piece, R_pulse, 'tab:blue', lw=1.5, alpha=0.85)
dot_rpulse, = ax_rpulse.plot([t_piece[0]], [R_pulse[0]], 'ro', ms=7, zorder=4)
vline_rpulse = ax_rpulse.axvline(t_piece[0], color='red', linestyle='--', alpha=0.6)
ax_rpulse.set_ylim(-0.05, 1.05)
ax_rpulse.set_ylabel(r"$R_{\mathrm{pulse}}(t)$")
ax_rpulse.set_xlabel("Time")
ax_rpulse.grid(True, alpha=0.3)

# --- State & Live Update Logic ---
current_t = 0
running = True

def update_visuals(idx):
    """Updates scatter points, 2D mean marker, and all 3 waveform time indicators."""
    # 1. Update circle
    xs_p, ys_p = get_points(idx)
    scat.set_offsets(np.c_[xs_p, ys_p])
    
    x_avg, y_avg = np.mean(xs_p), np.mean(ys_p)
    avg_dot_circle.set_data([x_avg], [y_avg])
    mean_line.set_data([0, x_avg], [0, y_avg])

    # 2. Update time series trackers
    t_val = t_piece[idx]
    
    dot_wave.set_data([t_val], [total_x[idx]])
    vline_wave.set_xdata([t_val, t_val])
    
    dot_rml.set_data([t_val], [R_ML[idx]])
    dot_rml_avg.set_data([t_val], [R_ML_running_avg[idx]])
    vline_rml.set_xdata([t_val, t_val])
    
    dot_rpulse.set_data([t_val], [R_pulse[idx]])
    vline_rpulse.set_xdata([t_val, t_val])


def update(frame):
    global current_t
    if running:
        current_t = (current_t + 10) % T

        slider.eventson = False
        slider.set_val(t_piece[current_t])
        slider.eventson = True

    update_visuals(current_t)
    return scat, avg_dot_circle, mean_line, dot_wave, vline_wave, dot_rml, dot_rml_avg, vline_rml, dot_rpulse, vline_rpulse


# --- Interactive Controls ---
ax_slider_thresh = plt.axes([0.15, 0.10, 0.55, 0.03])
slider_thresh = Slider(ax_slider_thresh, 'ML Threshold', 0.0, 1.0, valinit=0.8, color='orange')

ax_slider = plt.axes([0.15, 0.05, 0.55, 0.03])
slider = Slider(ax_slider, 'Time', t_piece[0], t_piece[-1], valinit=t_piece[0])

ax_button = plt.axes([0.78, 0.055, 0.1, 0.06])
button = Button(ax_button, 'Pause')


# --- Highlight Region Logic ---
highlight_collection = None

def update_highlights(val):
    """Fills the background of ax_wave where R_ML_running_avg >= threshold"""
    global highlight_collection
    
    # Remove the old highlighted region if it exists
    if highlight_collection is not None:
        highlight_collection.remove()
    
    # Determine which points meet the threshold condition
    condition = R_ML_running_avg >= val
    
    # Fill background on ax_wave. Using get_xaxis_transform() lets us use y=0 to y=1 
    # to fill the entire vertical height regardless of the data scale.
    highlight_collection = ax_wave.fill_between(
        t_piece, 0, 1, where=condition, 
        transform=ax_wave.get_xaxis_transform(), 
        color='orange', alpha=0.3, zorder=0
    )
    
    # Update the visual dashed line on the R_ML plot
    hline_thresh.set_ydata([val, val])
    fig.canvas.draw_idle()

# Connect the slider and initialize
slider_thresh.on_changed(update_highlights)
update_highlights(0.8)


def on_slider(val):
    global current_t
    current_t = np.searchsorted(t_piece, val)
    current_t = np.clip(current_t, 0, T - 1)
    update_visuals(current_t)
    fig.canvas.draw_idle()

slider.on_changed(on_slider)

def toggle(event):
    global running
    running = not running
    button.label.set_text('Play' if not running else 'Pause')

button.on_clicked(toggle)

# --- Start Animation ---
ani = FuncAnimation(fig, update, interval=50, blit=False)
print("Showing animation")
plt.show()