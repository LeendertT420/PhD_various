import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, jn_zeros
from matplotlib.widgets import Slider

# -----------------------------
# Parameters
# -----------------------------
R = 1.0
sigma = 0.04

Nr = 300
Ntheta = 220

n_modes = 100

T_max = 6  # maximum time to sweep
c3 = 1.0    # arbitrary third-sound speed for example

# -----------------------------
# Frequency dispersion (linear)
# -----------------------------
zeros = jn_zeros(0, n_modes)

def omega(n):
    k = zeros[n]/R
    return c3 * k

# -----------------------------
# Grid
# -----------------------------
r = np.linspace(0, R, Nr)
theta = np.linspace(0, 2*np.pi, Ntheta)

Rgrid, Tgrid = np.meshgrid(r, theta, indexing="ij")
X = Rgrid * np.cos(Tgrid)
Y = Rgrid * np.sin(Tgrid)

dr = r[1] - r[0]

# -----------------------------
# Initial radial Gaussian rim
# -----------------------------
f = np.exp(-((r-0.925*R)**4)/(2*sigma**4))

# -----------------------------
# Bessel basis
# -----------------------------
coeff = np.zeros(n_modes)
radial_modes = []

for n in range(n_modes):
    kr = zeros[n]/R
    mode = jv(0, kr*r)
    norm = np.sqrt(np.sum(mode**2 * r) * dr)
    mode /= norm
    radial_modes.append(mode)
    coeff[n] = np.sum(f * mode * r) * dr

radial_modes = np.array(radial_modes)
mode_max = np.max(np.abs(coeff[:,None]*radial_modes))

# -----------------------------
# Figure layout
# -----------------------------
fig = plt.figure(figsize=(13,5))
plt.subplots_adjust(bottom=0.2)

ax_disk = fig.add_subplot(131)
ax_radial = fig.add_subplot(132)
ax_modes = fig.add_subplot(133)

# 2D disk field
field = np.zeros_like(Rgrid)
im = ax_disk.pcolormesh(
    X, Y, field, shading="auto", cmap="viridis",
    vmin=-np.max(np.abs(f)), vmax=np.max(np.abs(f))
)
ax_disk.set_aspect("equal")
ax_disk.set_title("Disk field")

# radial profile
line_total, = ax_radial.plot([], [], lw=3, label="total")
line_initial, = ax_radial.plot(r, f, "--", lw=2, label="initial")
ax_radial.set_xlim(0, R)
ax_radial.set_ylim(-1.5*np.max(f), 1.5*np.max(f))
ax_radial.set_xlabel("r")
ax_radial.set_title("Radial profile")
ax_radial.legend()

# individual modes
mode_lines = []
for n in range(n_modes):
    line, = ax_modes.plot([], [], alpha=0.3)
    mode_lines.append(line)
ax_modes.set_xlim(0, R)
ax_modes.set_ylim(-1.2*mode_max, 1.2*mode_max)
ax_modes.set_xlabel("r")
ax_modes.set_title("Individual Bessel modes")

# timestamp
time_text = fig.suptitle("")

# -----------------------------
# Update function
# -----------------------------
def update(t):
    radial_total = np.zeros_like(r)
    components = []
    for n in range(n_modes):
        comp = coeff[n]*radial_modes[n]*np.cos(omega(n)*t)
        radial_total += comp
        components.append(comp)

    # update 2D field
    field = radial_total[:,None]*np.ones((1, Ntheta))
    im.set_array(field.ravel())

    # radial profile
    line_total.set_data(r, radial_total)

    # individual modes
    for n, line in enumerate(mode_lines):
        line.set_data(r, components[n])

    time_text.set_text(f"time = {t:.2f}")

    fig.canvas.draw_idle()

# -----------------------------
# Slider setup
# -----------------------------
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03], facecolor='lightgray')
slider = Slider(ax_slider, 'Time', 0.0, T_max, valinit=0.0)

def slider_update(val):
    update(slider.val)

slider.on_changed(slider_update)

# Initial draw
update(0.0)
plt.show()