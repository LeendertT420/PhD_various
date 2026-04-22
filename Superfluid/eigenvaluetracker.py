import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def x_star(alpha, delta):
    roots = np.roots([1, 2*delta, delta**2 + 1, -1*alpha])
    roots[~np.isreal(roots)] = np.nan
    return np.real(roots)

def L_prime(x, alpha, delta):
    return -2*alpha*(x+delta) / ((x+delta)**2+1)**2

# Define your matrix
def Jacobian(alpha, delta, zeta, tau):
    roots = x_star(alpha*tau, delta)
    roots = roots[~np.isnan(roots)]   # keep real ones

    J_list = []
    for x in roots:
        dLdt = L_prime(x, alpha, delta)
        J = np.array([[0   , 1      , 0     ],
                      [-1  , -2*zeta, 1     ],
                      [dLdt, 0      , -1/tau]])
        J_list.append(J)

    return J_list   # <-- changed (was single matrix)

def compute_tracked_eigs(alpha_vals, delta, zeta, tau):
    all_branches = []  # <-- NEW

    for a in alpha_vals:
        J_list = Jacobian(a, delta, zeta, tau)
        eigs_step = [np.linalg.eigvals(J) for J in J_list]
        all_branches.append(eigs_step)

    max_branches = max(len(step) for step in all_branches)

    # pad with NaNs so array is rectangular
    eigs_all = np.full((len(alpha_vals), max_branches, 3), np.nan, dtype=complex)

    for i, step in enumerate(all_branches):
        for j, eigvals in enumerate(step):
            eigs_all[i, j, :] = eigvals

    # track each branch separately
    tracked_all = []

    for b in range(max_branches):
        eigs_branch = eigs_all[:, b, :]

        tracked = np.zeros_like(eigs_branch, dtype=complex)
        tracked[0] = eigs_branch[0]

        for i in range(1, len(alpha_vals)):
            prev = tracked[i-1]
            curr = eigs_branch[i]

            remaining = list(curr)
            for j in range(3):
                distances = [abs(prev[j] - r) for r in remaining]
                idx = np.argmin(distances)
                tracked[i, j] = remaining.pop(idx)

        tracked_all.append(tracked)

    return tracked_all

alpha_vals = np.linspace(0, 10, 300)

delta0, zeta0, tau0 = -4, 0.05, 1
eigs = compute_tracked_eigs(alpha_vals, delta0, zeta0, tau0)

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.3)

colors = ['r', 'b', 'g']  # one per branch

tracked_all = compute_tracked_eigs(alpha_vals, delta0, zeta0, tau0)

lines = []

for b, eigs in enumerate(tracked_all):
    branch_lines = []
    for i in range(3):
        line, = ax.plot(eigs[:, i].real, eigs[:, i].imag,
                        lw=2, color=colors[b], label=f'branch {b+1}' if i==0 else "")
        branch_lines.append(line)
    lines.append(branch_lines)

# legend OUTSIDE
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

ax.set_xlabel("Re(λ)")
ax.set_ylabel("Im(λ)")
ax.set_title("Eigenvalue trajectories in complex plane")
ax.axhline(0, color='gray', lw=0.5)
ax.axvline(0, color='gray', lw=0.5)
ax.set_aspect('equal', adjustable='datalim')

ax_delta = plt.axes([0.2, 0.2, 0.65, 0.03])
ax_zeta = plt.axes([0.2, 0.15, 0.65, 0.03])
ax_tau = plt.axes([0.2, 0.1, 0.65, 0.03])

slider_delta = Slider(ax_delta, 'δ', -5, 0, valinit=delta0)
slider_zeta = Slider(ax_zeta, 'ζ', 0, 5, valinit=zeta0)
slider_tau = Slider(ax_tau, 'τ', 0, 5, valinit=tau0)

def update(val):
    delta = slider_delta.val
    zeta = slider_zeta.val
    tau = slider_tau.val

    tracked_all = compute_tracked_eigs(alpha_vals, delta, zeta, tau)

    for b, eigs in enumerate(tracked_all):
        for i in range(3):
            lines[b][i].set_data(eigs[:, i].real, eigs[:, i].imag)

    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()

slider_delta.on_changed(update)
slider_zeta.on_changed(update)
slider_tau.on_changed(update)

plt.show()
