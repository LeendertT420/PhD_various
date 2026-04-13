import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# -----------------------------
# Constants (set these!)
# -----------------------------
d = 1.0        # grating spacing (arbitrary units)
wavelength = 0.2  # wavelength (same units as d)


# -----------------------------
# Intensity function (Casini & Nelson)
# -----------------------------
def I(beta, n, alpha, phi):
    return np.sinc(
        n * np.pi * np.cos(alpha) / np.cos(alpha - phi) *
        (np.cos(phi) - np.sin(phi) / np.tan((alpha + beta) / 2))
    )**2


# -----------------------------
# Diffraction orders
# -----------------------------
def diffraction_orders(alpha, max_order=10):
    orders = []
    for m in range(-max_order, max_order + 1):
        val = m * wavelength / d + np.sin(alpha)

        # Only valid if |sin(beta)| ≤ 1
        if abs(val) <= 1:
            beta_m = np.arcsin(val)
            orders.append(beta_m)

    return orders


# -----------------------------
# Beta range
# -----------------------------
beta = np.linspace(-np.pi/2 + 0.01, np.pi/2 - 0.01, 1000)


# -----------------------------
# Initial parameters
# -----------------------------
alpha0 = np.deg2rad(20)
phi0   = np.deg2rad(30)
n0     = 5


# -----------------------------
# Plot setup
# -----------------------------
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.3)

line, = ax.plot(beta, I(beta, n0, alpha0, phi0), label="Blaze envelope")

# initial diffraction lines
order_lines = []
for b in diffraction_orders(alpha0):
    l = ax.axvline(b, linestyle='--')
    order_lines.append(l)

ax.set_xlabel("beta (radians)")
ax.set_ylabel("Intensity")
ax.set_title("Blazed Grating + Diffraction Orders")
ax.grid(True)


# -----------------------------
# Sliders
# -----------------------------
ax_alpha = plt.axes([0.2, 0.2, 0.65, 0.03])
ax_phi   = plt.axes([0.2, 0.15, 0.65, 0.03])
ax_n     = plt.axes([0.2, 0.1, 0.65, 0.03])

slider_alpha = Slider(ax_alpha, 'alpha (deg)', 0, 80, valinit=np.rad2deg(alpha0))
slider_phi   = Slider(ax_phi,   'phi (deg)',   0, 80, valinit=np.rad2deg(phi0))
slider_n     = Slider(ax_n,     'n',           1, 20, valinit=n0, valstep=1)


# -----------------------------
# Update function
# -----------------------------
def update(val):
    global order_lines

    alpha = np.deg2rad(slider_alpha.val)
    phi   = np.deg2rad(slider_phi.val)
    n     = slider_n.val

    # Update intensity curve
    y = I(beta, n, alpha, phi)
    line.set_ydata(y)

    # Remove old diffraction lines
    for l in order_lines:
        l.remove()
    order_lines = []

    # Draw new diffraction order lines
    for b in diffraction_orders(alpha):
        l = ax.axvline(b, linestyle='--')
        order_lines.append(l)

    fig.canvas.draw_idle()


slider_alpha.on_changed(update)
slider_phi.on_changed(update)
slider_n.on_changed(update)


plt.show()