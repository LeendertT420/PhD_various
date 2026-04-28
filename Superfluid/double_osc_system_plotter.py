import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle
from scipy.integrate import solve_ivp
from matplotlib.gridspec import GridSpec

from double_osc_eqs import *

# -----------------------------
# parameter ranges
# -----------------------------
delta_min, delta_max = -5, 2
deltas = np.linspace(delta_min, delta_max, 200)

alpha_min, alpha_max = 0, 5

# -----------------------------
# initial parameters
# -----------------------------
alpha0, delta0 = 0.0, 0.0
tau0 = 1.0
w10, w20 = 1.0, 1.0
g10, g20 = 0.05, 0.05

# ICs
x10, v10 = 0.0, 0.0
x20, v20 = 0.0, 0.0
z0 = 0.0
T0 = 100

colors = ['r', 'b', 'g', 'm']

# -----------------------------
# FIGURE
# -----------------------------
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(3, 3, hspace=0.4, wspace=0.3)

# LEFT COLUMN
ax1 = fig.add_subplot(gs[0, 0])  # bifurcation
ax2 = fig.add_subplot(gs[1, 0])  # eigenvalues

# CENTER COLUMN
ax3 = fig.add_subplot(gs[0, 1], projection='3d')  # (x1,v1,z)
ax4 = fig.add_subplot(gs[1, 1], projection='3d')  # (x2,v2,z)
ax5 = fig.add_subplot(gs[2, 1])                  # (x1,x2)

# RIGHT COLUMN (time series)
ax6 = fig.add_subplot(gs[0, 2])
ax7 = fig.add_subplot(gs[1, 2])

# =============================
# BIFURCATION
# =============================
ax1.fill_between(deltas, alpha_min, alpha_max, color='lightblue', alpha=0.3)

bif_lower, = ax1.plot([], [], 'k', lw=2)
bif_upper, = ax1.plot([], [], 'k', lw=2)
lasing_line1, = ax1.plot([], [], 'r', lw=1)
lasing_line2, = ax1.plot([], [], 'r', lw=1)
lasing_line3, = ax1.plot([], [], 'r', lw=1)
lasing_lines = [lasing_line1, lasing_line2, lasing_line3]

point, = ax1.plot([], [], 'ko')

ax1.set_xlim(delta_min, delta_max)
ax1.set_ylim(alpha_min, alpha_max)
ax1.set_xlabel(r'$\delta$')
ax1.set_ylabel(r'$\alpha$')
ax1.set_title("Bifurcation diagram")

# =============================
# EIGENVALUES
# =============================
scatters = [ax2.scatter([], [], color=c) for c in colors]

ax2.axhline(0, color='gray')
ax2.axvline(0, color='gray')
ax2.set_xlim(-3, 3)
ax2.set_ylim(-3, 3)
ax2.set_title("Eigenvalues")

# =============================
# PHASE SPACE
# =============================
traj1, = ax3.plot([], [], [], 'k')
traj2, = ax4.plot([], [], [], 'k')

traj12, = ax5.plot([], [], 'k')

ax3.set_title("(x1, v1, z)")
ax4.set_title("(x2, v2, z)")
ax5.set_title("(x1, x2)")

# =============================
# TIME SERIES
# =============================
line_x1, = ax6.plot([], [], label='x1')
line_v1, = ax6.plot([], [], label='v1')
line_z1, = ax6.plot([], [], label='z')

ax6.set_title("Oscillator 1")

line_x2, = ax7.plot([], [], label='x2')
line_v2, = ax7.plot([], [], label='v2')

ax7.set_title("Oscillator 2")

for ax in [ax6, ax7]:
    ax.legend()

# =============================
# SLIDERS
# =============================
box = Rectangle((0.55, 0.02), 0.4, 0.25,
                transform=fig.transFigure,
                fill=False, linewidth=2)
fig.patches.append(box)

def make_slider(x, y, label, vmin, vmax, vinit):
    ax = fig.add_axes([x, y, 0.15, 0.02])
    return Slider(ax, label, vmin, vmax, valinit=vinit)

sA = make_slider(0.57, 0.23, 'α', 0, 5, alpha0)
sD = make_slider(0.75, 0.23, 'δ', delta_min, delta_max, delta0)
sT = make_slider(0.57, 0.19, 'τ', 0.1, 5, tau0)

sW1 = make_slider(0.75, 0.19, 'ω1', 0.1, 3, w10)
sW2 = make_slider(0.57, 0.15, 'ω2', 0.1, 3, w20)

sG1 = make_slider(0.75, 0.15, 'γ1', 0, 2, g10)
sG2 = make_slider(0.57, 0.11, 'γ2', 0, 2, g20)

sX10 = make_slider(0.75, 0.11, 'x1₀', -5, 5, x10)
sX20 = make_slider(0.57, 0.07, 'x2₀', -5, 5, x20)

sTime = make_slider(0.75, 0.07, 'T', 1, 100, T0)

# =============================
# UPDATE
# =============================
def update(val):

    alpha, delta = sA.val, sD.val
    tau = sT.val
    w1, w2 = sW1.val, sW2.val
    g1, g2 = sG1.val, sG2.val

    T = sTime.val

    # bifurcation
    bif_lower.set_data(deltas, lower_boundary(deltas, w1, w2)*rho(w1, w2))
    bif_upper.set_data(deltas, upper_boundary(deltas, w1, w2)*rho(w1, w2))

    thresholds = lasing_threshold(deltas, tau, w1, w2, g1, g2)
    print(np.shape(thresholds))
    for threshold, lasing_line in zip(thresholds, lasing_lines):
        lasing_line.set_data(deltas, threshold*rho(w1, w2))

    point.set_data([delta], [alpha*rho(w1, w2)])

    # eigenvalues
    eigs_all = compute_eigs(alpha, delta, tau, w1, w2, g1, g2)

    for i in range(4):
        if i < len(eigs_all):
            eigs = eigs_all[i]
            scatters[i].set_offsets(np.c_[eigs.real, eigs.imag])
        else:
            scatters[i].set_offsets(np.empty((0,2)))

    # solve system
    y0 = [sX10.val, 0, sX20.val, 0, z0]

    t_eval = np.linspace(0, T, 1500)

    sol = solve_ivp(
        lambda t,y: system(t, y, alpha, delta, tau, w1, w2, g1, g2),
        (0, T), y0, t_eval=t_eval
    )

    x1, v1, x2, v2, z = sol.y

    # trajectories
    traj1.set_data(x1, v1)
    traj1.set_3d_properties(z)

    traj2.set_data(x2, v2)
    traj2.set_3d_properties(z)

    traj12.set_data(x1, x2)

    # autoscale
    for ax, data in zip([ax3, ax4],
                       [(x1,v1,z), (x2,v2,z)]):

        mins = [d.min() for d in data]
        maxs = [d.max() for d in data]
        m = [0.1*(maxs[i]-mins[i]+1e-6) for i in range(3)]

        ax.set_xlim(mins[0]-m[0], maxs[0]+m[0])
        ax.set_ylim(mins[1]-m[1], maxs[1]+m[1])
        ax.set_zlim(mins[2]-m[2], maxs[2]+m[2])

    ax5.set_xlim(x1.min(), x1.max())
    ax5.set_ylim(x2.min(), x2.max())

    # time series
    line_x1.set_data(t_eval, x1)
    line_v1.set_data(t_eval, v1)
    line_z1.set_data(t_eval, z)

    line_x2.set_data(t_eval, x2)
    line_v2.set_data(t_eval, v2)

    for ax in [ax6, ax7]:
        ax.set_xlim(0, T)

    ax6.set_ylim(min(x1.min(), v1.min(), z.min()),
                 max(x1.max(), v1.max(), z.max()))

    ax7.set_ylim(min(x2.min(), v2.min()),
                 max(x2.max(), v2.max()))

    fig.canvas.draw_idle()

# connect sliders
for s in [sA,sD,sT,sW1,sW2,sG1,sG2,sX10,sX20,sTime]:
    s.on_changed(update)

update(None)
plt.show()