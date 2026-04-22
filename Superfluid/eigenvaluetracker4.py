import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# -----------------------------
# delta grid
# -----------------------------
deltas = np.linspace(-10, 0, 2000)

# -----------------------------
# bifurcation boundaries
# -----------------------------
def lower_boundary(d):
    s = np.sqrt(d**2 - 3)
    return -2/27 * (s - 2*d)**2 * (s + d)

def upper_boundary(d):
    s = np.sqrt(d**2 - 3)
    return  2/27 * (s + 2*d)**2 * (s - d)

# -----------------------------
# lasing threshold
# -----------------------------
def lasing_threshold(d, zeta, tau):
    B = zeta*(2*zeta*tau + tau + 1/tau)
    x = ((1 - 2*B)*d - np.sqrt(d**2 - 4*B*(B - 1))) / (2*B - 2)
    return x * ((x + d)**2 + 1)

# -----------------------------
# fixed points
# -----------------------------
def x_star(alpha, delta):
    roots = np.roots([1, 2*delta, delta**2 + 1, -alpha])
    roots = np.real(roots[np.isreal(roots)])
    return roots

def L_prime(x, alpha, delta):
    return -2*alpha*(x + delta) / ((x + delta)**2 + 1)**2

# -----------------------------
# Jacobian
# -----------------------------
def Jacobian(alpha, delta, zeta, tau):
    roots = x_star(alpha, delta)

    J_list = []
    for x in roots:
        dLdt = L_prime(x, alpha, delta)
        J = np.array([
            [0,    1,        0],
            [-1,  -2*zeta,   1],
            [dLdt/tau, 0,   -1/tau]
        ])
        J_list.append(J)

    return J_list

def compute_eigs(alpha, delta, zeta, tau):
    return [np.linalg.eigvals(J) for J in Jacobian(alpha, delta, zeta, tau)]

# -----------------------------
# initial params
# -----------------------------
alpha0, delta0, zeta0, tau0 = 1.0, -4.0, 0.05, 1.0

# -----------------------------
# FIGURE (2 PANELS)
# -----------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
plt.subplots_adjust(bottom=0.35)

# =============================
# LEFT: bifurcation diagram
# =============================
ax1.set_xlim(deltas.min(), 0)
ax1.set_ylim(0, 10)
ax1.set_xlabel(r'$\delta$')
ax1.set_ylabel(r'$\alpha$')
ax1.set_title("Bifurcation diagram")

ax1.fill_between(deltas, 0, 10, color='lightblue', alpha=0.3)
ax1.fill_between(deltas, lower_boundary(deltas), upper_boundary(deltas),
                 color='orange', alpha=0.5)

lower_line, = ax1.plot(deltas, lower_boundary(deltas), 'k', lw=2, label='Bifurcation boundary')
upper_line, = ax1.plot(deltas, upper_boundary(deltas), 'k', lw=2)
lasing_line, = ax1.plot([], [], 'k--', lw=2, label='Lasing threshold')

ax1.legend()

# current position marker
point, = ax1.plot([], [], 'ro', markersize=8, color='k')

# =============================
# RIGHT: eigenvalues
# =============================
colors = ['r', 'b', 'g']
scatters = [ax2.scatter([], [], color=c) for c in colors]

ax2.axhline(0, color='gray', lw=0.5)
ax2.axvline(0, color='gray', lw=0.5)

ax2.set_xlabel("Re(λ)")
ax2.set_ylabel("Im(λ)")
ax2.set_title("Eigenvalues")

ax2.set_xlim(-2, 2)
ax2.set_ylim(-2, 2)

legend = ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

# =============================
# SLIDERS
# =============================
ax_alpha = plt.axes([0.2, 0.25, 0.5, 0.03])
ax_delta = plt.axes([0.2, 0.2, 0.5, 0.03])
ax_zeta  = plt.axes([0.2, 0.15, 0.5, 0.03])
ax_tau   = plt.axes([0.2, 0.1, 0.5, 0.03])

sA = Slider(ax_alpha, 'α', 0, 10, valinit=alpha0)
sD = Slider(ax_delta, 'δ', -10, 0, valinit=delta0)
sZ = Slider(ax_zeta,  'ζ', 0, 2, valinit=zeta0)
sT = Slider(ax_tau,   'τ', 0, 5, valinit=tau0)

# =============================
# UPDATE
# =============================
def update(val):
    alpha = sA.val
    delta = sD.val
    zeta  = sZ.val
    tau   = sT.val

    # ---- bifurcation curve ----
    lasing = lasing_threshold(deltas, zeta, tau)
    lasing_line.set_data(deltas, lasing)

    # ---- current point ----
    point.set_data([delta], [alpha])

    # ---- eigenvalues ----
    eigs_all = compute_eigs(alpha, delta, zeta, tau)

    labels = []

    for i in range(3):
        if i < len(eigs_all):
            eigs = eigs_all[i]
            pts = np.column_stack((eigs.real, eigs.imag))

            # corresponding fixed point
            roots = x_star(alpha, delta)
            x_val = roots[i] if i < len(roots) else np.nan

            labels.append(f"x* = {x_val:.3f}")
        else:
            pts = np.empty((0, 2))
            labels.append("")

        scatters[i].set_offsets(pts)
        scatters[i].set_label(labels[i])

    # refresh legend
    ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

    fig.canvas.draw_idle()

# connect sliders
for s in [sA, sD, sZ, sT]:
    s.on_changed(update)

update(None)
plt.show()