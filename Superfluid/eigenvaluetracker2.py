import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# -----------------------------
# Root computation
# -----------------------------
def x_star(alpha, delta):
    roots = np.roots([1, 2*delta, delta**2 + 1, -alpha])
    roots = np.real(roots[np.isreal(roots)])  # keep only real roots
    return roots

def L_prime(x, alpha, delta):
    return -2*alpha*(x + delta) / ((x + delta)**2 + 1)**2

# -----------------------------
# Jacobian (returns list of matrices)
# -----------------------------
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

# -----------------------------
# Eigenvalue computation
# -----------------------------
def compute_eigs(alpha, delta, zeta, tau):
    J_list = Jacobian(alpha, delta, zeta, tau)

    eigs_all = []
    for J in J_list:
        eigs_all.append(np.linalg.eigvals(J))

    return eigs_all  # list of (3,) arrays


# -----------------------------
# Initial parameters
# -----------------------------
alpha0, delta0, zeta0, tau0 = 1.0, -4.0, 0.05, 1.0

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.35, right=0.75)

colors = ['r', 'b', 'g']

# Create fixed scatter objects (max 3 branches)
scatters = []
for b in range(3):
    sc = ax.scatter([], [], color=colors[b], label=f'branch {b+1}')
    scatters.append(sc)

ax.set_xlabel("Re(λ)")
ax.set_ylabel("Im(λ)")
ax.set_title("Eigenvalues in complex plane")
ax.axhline(0, color='gray', lw=0.5)
ax.axvline(0, color='gray', lw=0.5)
ax.set_aspect('equal', adjustable='datalim')

ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

# -----------------------------
# Sliders
# -----------------------------
ax_alpha = plt.axes([0.2, 0.25, 0.5, 0.03])
ax_delta = plt.axes([0.2, 0.2, 0.5, 0.03])
ax_zeta  = plt.axes([0.2, 0.15, 0.5, 0.03])
ax_tau   = plt.axes([0.2, 0.1, 0.5, 0.03])

slider_alpha = Slider(ax_alpha, 'α', 0.1, 10, valinit=alpha0)
slider_delta = Slider(ax_delta, 'δ', -5, 0, valinit=delta0)
slider_zeta  = Slider(ax_zeta,  'ζ', 0, 5, valinit=zeta0)
slider_tau   = Slider(ax_tau,   'τ', 0.1, 5, valinit=tau0)

# -----------------------------
# Update function
# -----------------------------
def update(val):
    alpha = slider_alpha.val
    delta = slider_delta.val
    zeta  = slider_zeta.val
    tau   = slider_tau.val

    eigs_all = compute_eigs(alpha, delta, zeta, tau)

    all_points = []

    for b in range(3):
        if b < len(eigs_all):
            eigs = eigs_all[b]
            points = np.column_stack((eigs.real, eigs.imag))
        else:
            points = np.empty((0, 2))

        scatters[b].set_offsets(points)

        if points.size > 0:
            all_points.append(points)

    # Stable axis scaling
    if all_points:
        all_points = np.vstack(all_points)
        ax.set_xlim(all_points[:, 0].min() - 0.5,
                    all_points[:, 0].max() + 0.5)
        ax.set_ylim(all_points[:, 1].min() - 0.5,
                    all_points[:, 1].max() + 0.5)
    
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    fig.canvas.draw_idle()

# connect sliders
slider_alpha.on_changed(update)
slider_delta.on_changed(update)
slider_zeta.on_changed(update)
slider_tau.on_changed(update)

# IMPORTANT: initial draw
update(None)

plt.show()