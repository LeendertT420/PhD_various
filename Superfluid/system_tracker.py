import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle
from scipy.integrate import solve_ivp
from matplotlib.gridspec import GridSpec
from lagged_lasing import *

# -----------------------------
# delta grid
# -----------------------------
delta_min, delta_max = -5, 2
deltas = np.linspace(delta_min, delta_max, 100)

alpha_min, alpha_max = 0, 5

# -----------------------------
# initial params
# -----------------------------
alpha0, delta0, zeta0, tau0 = 1.0, -2.0, 0.05, 1.0
x0, v0, z0 = 0.5, 0.0, 0.0
T0 = 20

colors = ['r', 'b', 'g']

# -----------------------------
# FIGURE + GRID
# -----------------------------
fig = plt.figure(figsize=(14, 9))
gs = GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1],
              hspace=0.4, wspace=0.3)

# LEFT
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[2, 0])

# RIGHT
ax3 = fig.add_subplot(gs[0:2, 1], projection='3d')

# =============================
# BIFURCATION
# =============================
ax1.fill_between(deltas, alpha_min, alpha_max, color='lightblue', alpha=0.3)
ax1.fill_between(deltas, lower_boundary(deltas), upper_boundary(deltas),
                 color='orange', alpha=0.5)

ax1.plot(deltas, lower_boundary(deltas), 'k', lw=2, label='Bifurcation boundary')
ax1.plot(deltas, upper_boundary(deltas), 'k', lw=2)
lasing_line, = ax1.plot([], [], 'k--', lw=2, label='Lasing threshold')

point, = ax1.plot([], [], 'ko')

ax1.set_xlim(delta_min, delta_max)
ax1.set_ylim(alpha_min, alpha_max)
ax1.set_xlabel(r'$\delta$')
ax1.set_ylabel(r'$\alpha$')
ax1.set_title("Bifurcation diagram")
ax1.legend()

# =============================
# EIGENVALUES
# =============================
scatters = [ax2.scatter([], [], color=c) for c in colors]

ax2.axhline(0, color='gray')
ax2.axvline(0, color='gray')

ax2.set_xlim(-2, 2)
ax2.set_ylim(-2, 2)
ax2.set_xlabel("Re(λ)")
ax2.set_ylabel("Im(λ)")
ax2.set_title("Eigenvalues")

# =============================
# PHASE SPACE
# =============================
traj_line, = ax3.plot([], [], [], 'k')
init_point = ax3.scatter([], [], [], color='k')
fp_scatter = [ax3.scatter([], [], [], color=c, s=50) for c in colors]

ax3.set_xlabel("x")
ax3.set_ylabel("v")
ax3.set_zlabel("z")
ax3.set_title("Phase space")

# =============================
# TIME SERIES
# =============================
line_x, = ax4.plot([], [], label='x')
line_v, = ax4.plot([], [], label='v')
line_z, = ax4.plot([], [], label='z')

ax4.set_xlabel("t")
ax4.set_ylabel("state")
ax4.set_title("Time series")
ax4.legend()

# =============================
# SLIDERS (figure-aligned box)
# =============================
# Box in figure coords (FIXED)
box = Rectangle((0.55, 0.05), 0.4, 0.22,
                transform=fig.transFigure,
                fill=False, linewidth=2)
fig.patches.append(box)

fig.text(0.56, 0.26, "Controls", fontsize=11)

def make_slider(x, y, label, vmin, vmax, vinit):
    ax = fig.add_axes([x, y, 0.15, 0.02])
    return Slider(ax, label, vmin, vmax, valinit=vinit)

# parameters
sA = make_slider(0.57, 0.23, 'α', 0, 5, alpha0)
sD = make_slider(0.75, 0.23, 'δ', delta_min, delta_max, delta0)
sZ = make_slider(0.57, 0.18, 'ζ', 0, 2, zeta0)
sT = make_slider(0.75, 0.18, 'τ', 0.1, 5, tau0)

# ICs
sX0 = make_slider(0.57, 0.13, r'$x_0$', -5, 5, x0)
sV0 = make_slider(0.75, 0.13, r'$y_0$', -5, 5, v0)
sZ0 = make_slider(0.57, 0.08, r'$z_0$', -5, 5, z0)
sTime = make_slider(0.75, 0.08, r'$T$', 1, 100, T0)

# =============================
# UPDATE
# =============================
def update(val):
    alpha, delta = sA.val, sD.val
    zeta, tau = sZ.val, sT.val
    x_init, v_init, z_init = sX0.val, sV0.val, sZ0.val
    T = sTime.val

    lasing_line.set_data(deltas, lasing_threshold(deltas, zeta, tau))
    point.set_data([delta], [alpha])

    eigs_all = compute_eigs(alpha, delta, zeta, tau)
    roots = x_star(alpha, delta)

    for i in range(3):
        if i < len(eigs_all):
            eigs = eigs_all[i]
            scatters[i].set_offsets(np.c_[eigs.real, eigs.imag])
            scatters[i].set_label(f"x*={roots[i]:.2f}")
        else:
            scatters[i].set_offsets(np.empty((0,2)))
            scatters[i].set_label("")
    ax2.legend()

    t_eval = np.linspace(0, T, 1000)
    sol = solve_ivp(lambda t,y: system(t,y,alpha,delta,zeta,tau),
                    (0,T), [x_init,v_init,z_init], t_eval=t_eval)

    # trajectory
    traj_line.set_data(sol.y[0], sol.y[1])
    traj_line.set_3d_properties(sol.y[2])
    init_point._offsets3d = ([x_init],[v_init],[z_init])

    # FIXED POINTS (added back correctly)
    for i, scat in enumerate(fp_scatter):
        if i < len(roots):
            scat._offsets3d = ([roots[i]], [0], [roots[i]])
        else:
            scat._offsets3d = ([], [], [])

    # autoscale 3D
    mins = sol.y.min(axis=1)
    maxs = sol.y.max(axis=1)
    m = 0.1*(maxs-mins+1e-6)

    ax3.set_xlim(mins[0]-m[0], maxs[0]+m[0])
    ax3.set_ylim(mins[1]-m[1], maxs[1]+m[1])
    ax3.set_zlim(mins[2]-m[2], maxs[2]+m[2])

    # time series
    line_x.set_data(t_eval, sol.y[0])
    line_v.set_data(t_eval, sol.y[1])
    line_z.set_data(t_eval, sol.y[2])

    ax4.set_xlim(0, T)
    ax4.set_ylim(sol.y.min(), sol.y.max())

    fig.canvas.draw_idle()

for s in [sA,sD,sZ,sT,sX0,sV0,sZ0,sTime]:
    s.on_changed(update)

update(None)
plt.show()