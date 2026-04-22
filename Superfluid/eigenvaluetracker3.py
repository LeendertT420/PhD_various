import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# -----------------------------
# delta grid
# -----------------------------
deltas = np.linspace(-10, 0, 2000)

# -----------------------------
# boundaries (fixed)
# -----------------------------
def lower_boundary(d):
    s = np.sqrt(d**2 - 3)
    return -2/27 * (s - 2*d)**2 * (s + d)

def upper_boundary(d):
    s = np.sqrt(d**2 - 3)
    return  2/27 * (s + 2*d)**2 * (s - d)

# -----------------------------
# lasing threshold (NOW DYNAMIC)
# -----------------------------
def lasing_threshold(d, zeta, tau):
    B = zeta*(2*zeta*tau + tau**2 + 1)
    x = ((1 - 2*B)*d - np.sqrt(d**2 - 4*B*(B - 1))) / (2*B - 2)
    return x/tau * ((x + d)**2 + 1)

# -----------------------------
# Jacobian part (unchanged)
# -----------------------------
def x_star(alpha, delta):
    roots = np.roots([1, 2*delta, delta**2 + 1, -alpha])
    roots = np.real(roots[np.isreal(roots)])
    return roots

def L_prime(x, alpha, delta):
    return -2*alpha*(x + delta) / ((x + delta)**2 + 1)**2

def Jacobian(alpha, delta, zeta, tau):
    roots = x_star(alpha * tau, delta)

    J_list = []
    for x in roots:
        dLdt = L_prime(x, alpha, delta)
        J = np.array([
            [0,    1,        0],
            [-1,  -2*zeta,   1],
            [dLdt, 0,   -1/tau]
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
# FIGURE: TWO PANELS
# -----------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plt.subplots_adjust(bottom=0.35)

# =============================
# PANEL 1: BIFURCATION
# =============================
lower_line, = ax1.plot([], [], 'k', lw=2)
upper_line, = ax1.plot([], [], 'k', lw=2)
lasing_line, = ax1.plot([], [], 'k--', lw=2)

ax1.set_xlim(deltas.min(), 0)
ax1.set_ylim(0, 10)
ax1.set_xlabel(r'$\delta$')
ax1.set_ylabel(r'$\alpha$')
ax1.set_title("Bifurcation diagram")

# shading (static background)
ax1.fill_between(deltas, 0, 10, color='lightblue', alpha=0.3)
ax1.fill_between(deltas, lower_boundary(deltas), upper_boundary(deltas),
                 color='orange', alpha=0.5)

# =============================
# PANEL 2: EIGENVALUES
# =============================
colors = ['r', 'b', 'g']
scatters = [ax2.scatter([], [], color=c) for c in colors]

ax2.axhline(0, color='gray', lw=0.5)
ax2.axvline(0, color='gray', lw=0.5)
ax2.set_xlabel("Re(λ)")
ax2.set_ylabel("Im(λ)")
ax2.set_title("Eigenvalues")

ax2.set_xlim(-10, 10)
ax2.set_ylim(-10, 10)

# =============================
# SLIDERS
# =============================
ax_alpha = plt.axes([0.2, 0.25, 0.5, 0.03])
ax_delta = plt.axes([0.2, 0.2, 0.5, 0.03])
ax_zeta  = plt.axes([0.2, 0.15, 0.5, 0.03])
ax_tau   = plt.axes([0.2, 0.1, 0.5, 0.03])

sA = Slider(ax_alpha, 'α', 0.1, 10, valinit=alpha0)
sD = Slider(ax_delta, 'δ', -5, 0, valinit=delta0)
sZ = Slider(ax_zeta,  'ζ', 0, 1, valinit=zeta0)
sT = Slider(ax_tau,   'τ', 0.1, 5, valinit=tau0)

# =============================
# UPDATE FUNCTION
# =============================
def update(val):
    alpha = sA.val
    delta = sD.val
    zeta  = sZ.val
    tau   = sT.val

    # ---- update bifurcation curve ----
    lasing = lasing_threshold(deltas, zeta, tau)

    lasing_line.set_data(deltas, lasing)
    lower_line.set_data(deltas, lower_boundary(deltas))
    upper_line.set_data(deltas, upper_boundary(deltas))

    # ---- update eigenvalues ----
    eigs_all = compute_eigs(alpha, delta, zeta, tau)

    for i in range(3):
        if i < len(eigs_all):
            eigs = eigs_all[i]
            pts = np.column_stack((eigs.real, eigs.imag))
        else:
            pts = np.empty((0, 2))

        scatters[i].set_offsets(pts)

    fig.canvas.draw_idle()

# connect sliders
for s in [sA, sD, sZ, sT]:
    s.on_changed(update)

update(None)
plt.show()