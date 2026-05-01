import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
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
rho0 = 1.0
gamma10, gamma20 = 0.05, 0.05

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
ax5 = fig.add_subplot(gs[2, 0])                  # (x1,x2)

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
scatters = [ax2.scatter([], [], color=c, s=10) for c in colors]

ax2.axhline(0, color='gray')
ax2.axvline(0, color='gray')
ax2.set_xlim(-1, 1)
ax2.set_ylim(-1.5, 1.5)
ax2.set_title("Eigenvalues")

# =============================
# PHASE SPACE
# =============================
traj1, = ax3.plot([], [], [], 'k')
traj2, = ax4.plot([], [], [], 'k')

traj12, = ax5.plot([], [], 'k')

ax3.set_title(r"$(x_+,\,v_+,\,z)$")
ax4.set_title(r"$(x_-,\,v_-,\,z)$")
ax5.set_title("(x1, x2)")

# =============================
# TIME SERIES
# =============================
line_xplus, = ax6.plot([], [], label=r'$x_+$')
line_vplus, = ax6.plot([], [], label=r'$v_+$')
line_z1, = ax6.plot([], [], label='z')

ax6.set_title("Oscillator 1")

line_xminus, = ax7.plot([], [], label=r'$x_-$')
line_vminus, = ax7.plot([], [], label=r'$v_-$')

ax7.set_title("Oscillator 2")

for ax in [ax6, ax7]:
    ax.legend()

# =============================
# SLIDERS
# =============================
box = Rectangle((0.45, 0.05), 0.5, 0.25,
                transform=fig.transFigure,
                fill=False, linewidth=2)
fig.patches.append(box)

def make_slider(x, y, label, vmin, vmax, vinit, step=None):
    ax = fig.add_axes([x, y, 0.15, 0.02])
    return Slider(ax, label, vmin, vmax, valinit=vinit, valstep=step)

def make_slider_with_box(x, y, label, vmin, vmax, vinit, step=None):
    # Slider
    ax_slider = fig.add_axes([x, y, 0.15, 0.02])
    slider = Slider(ax_slider, label, vmin, vmax, valinit=vinit, valstep=step)

    # Text box (to the right)
    ax_box = fig.add_axes([x + 0.17, y, 0.05, 0.03])
    text_box = TextBox(ax_box, '', initial=str(vinit))

    # When slider moves → update text
    def update_text(val):
        text_box.set_val(f"{val:.3f}")
    slider.on_changed(update_text)

    # When user types → update slider
    def submit(text):
        try:
            val = float(text)
            if vmin <= val <= vmax:
                slider.set_val(val)
        except ValueError:
            pass
    text_box.on_submit(submit)

    return slider, text_box

sA = make_slider(0.5, 0.23, r'$\alpha$', 0, 5, alpha0, step=.01)
sD = make_slider(0.75, 0.23, r'$\delta$', delta_min, delta_max, delta0, step=.01)
sT = make_slider(0.5, 0.19, r'$\tau$', 0.1, 5, tau0)

srho = make_slider(0.75, 0.19, r'$\rho$', 0.1, 3, rho0, step=.1)

sG1 = make_slider(0.75, 0.15, r'$\gamma_1$', 0, 2, gamma10, step=.01)
sG2 = make_slider(0.5, 0.15, r'$\gamma_2$', 0, 2, gamma20, step=.01)

sX10 = make_slider(0.75, 0.11, r'$x_{1,0}$', -5, 5, x10, step=.1)
sX20 = make_slider(0.5, 0.11, r'$x_{2,0}$', -5, 5, x20, step=.1)

sTime = make_slider(0.75, 0.07, r'$T$', 1, 250, T0, step=1)

# =============================
# UPDATE
# =============================
def update(val):

    alpha, delta = sA.val, sD.val
    tau = sT.val
    rho = srho.val
    g1, g2 = sG1.val, sG2.val

    T = sTime.val

    # bifurcation
    bif_lower.set_data(deltas, lower_boundary(deltas, rho)*sigma(rho))
    bif_upper.set_data(deltas, upper_boundary(deltas, rho)*sigma(rho))

    thresholds = lasing_threshold(deltas, tau, rho, g1, g2)

    for threshold, lasing_line in zip(thresholds, lasing_lines):
        lasing_line.set_data(deltas, threshold*sigma(rho))

    point.set_data([delta], [alpha*sigma(rho)])

    # eigenvalues
    roots, eigvals, eigvecs = compute_eigs(alpha, delta, tau, rho, g1, g2)

    if not hasattr(update, "eig_quivers"):
        update.eig_quivers = []
    else:
        for q in update.eig_quivers:
            q.remove()
        update.eig_quivers = []

    scale = 0.15

    # for i, root in enumerate(roots):
    #     vals = eigvals[i]
    #     vecs = eigvecs[i]
    #     for j, (l, v) in enumerate(zip(vals, vecs)):
    #         if np.isreal(l):
    #             v = np.real(v)

    #             vnorm =  rotate(v) / np.linalg.norm(v)

    #             q = ax3.quiver(
    #                 root, 0, root,
    #                 vnorm[2], vnorm[3], vnorm[4],
    #                 color=colors[i % len(colors)],
    #                 linewidth=2
    #             )
    #             update.eig_quivers.append(q)

    #             u = ax4.quiver(
    #                 root, 0, root,
    #                 vnorm[0], vnorm[1], vnorm[4],
    #                 color=colors[i % len(colors)],
    #                 linewidth=2
    #             )
    #             update.eig_quivers.append(upper_boundary)

    #         else:
    #             vreal = np.real(rotate(v))
    #             vimag = np.imag(rotate(v))

    #             vrnorm = scale * vreal# / np.linalg.norm(vreal)
    #             vinorm = scale * vimag# / np.linalg.norm(vimag)

    #             r = ax3.quiver(
    #                 root, 0, root,
    #                 vrnorm[2], vrnorm[3], vrnorm[4],
    #                 color=colors[i % len(colors)],
    #                 linewidth=2
    #             )
    #             update.eig_quivers.append(r)

    #             v = ax4.quiver(
    #                 root, 0, root,
    #                 vrnorm[0], vrnorm[1], vrnorm[4],
    #                 color=colors[i % len(colors)],
    #                 linewidth=2
    #             )
    #             update.eig_quivers.append(v)

    #             s = ax3.quiver(
    #                 root, 0, root,
    #                 vinorm[2], vinorm[3], vinorm[4],
    #                 color=colors[i % len(colors)],
    #                 linewidth=2
    #             )
    #             update.eig_quivers.append(s)

    #             w = ax4.quiver(
    #                 root, 0, root,
    #                 vinorm[0], vinorm[1], vinorm[4],
    #                 color=colors[i % len(colors)],
    #                 linewidth=2
    #             )
    #             update.eig_quivers.append(w)

    for i in range(4):
        if i < len(eigvals):
            eigs = eigvals[i]
            scatters[i].set_offsets(np.c_[eigs.real, eigs.imag])
        else:
            scatters[i].set_offsets(np.empty((0,2)))

    # solve system
    y0 = [sX10.val, 0, sX20.val, 0, z0]

    t_eval = np.linspace(0, T, 20*T)

    sol = solve_ivp(
        lambda t,y: system(t, y, alpha, delta, tau, rho, g1, g2),
        (0, T), y0, t_eval=t_eval
    )

    x1, v1, x2, v2, z = sol.y

    # trajectories
    traj1.set_data(x1+x2, v1+v2)
    traj1.set_3d_properties(z)

    traj2.set_data(x1-x2, v1-v2)
    traj2.set_3d_properties(z)

    traj12.set_data(x1, x2)

    # autoscale
    for ax, data in zip([ax3, ax4],
                       [(x1+x2,v1+v2,z), (x1-x2,v1-v2,z)]):

        mins = [d.min() for d in data]
        maxs = [d.max() for d in data]
        m = [0.1*(maxs[i]-mins[i]+1e-6) for i in range(3)]

        ax.set_xlim(mins[0]-m[0], maxs[0]+m[0])
        ax.set_ylim(mins[1]-m[1], maxs[1]+m[1])
        ax.set_zlim(mins[2]-m[2], maxs[2]+m[2])

    ax5.set_xlim((x1+x2).min(), (x1+x2).max())
    ax5.set_ylim((x1-x2).min(), (x1-x2).max())

    # time series
    line_xplus.set_data(t_eval, x1+x2)
    line_vplus.set_data(t_eval, v1+v2)
    line_z1.set_data(t_eval, z)

    line_xminus.set_data(t_eval, x1-x2)
    line_vminus.set_data(t_eval, v1-v2)

    for ax in [ax6, ax7]:
        ax.set_xlim(0, T)

    limits_constant = True
    if limits_constant:
        ax6.set_ylim(-5,5)

        ax7.set_ylim(-5, 5)
    else:
        ax6.set_ylim(min((x1+x2).min(), (v1+v2).min(), z.min()),
                    max((x1+x2).max(), (v1+v2).max(), z.max()))

        ax7.set_ylim(min((x1-x2).min(), (v1-v2).min()),
                    max((x1-x2).max(), (v1-v2).max()))
        
    ax6.grid(True)
    ax7.grid(True)

    fig.canvas.draw_idle()

# connect sliders
for s in [sA,sD,sT,srho,sG1,sG2,sX10,sX20,sTime]:
    s.on_changed(update)

update(None)
plt.show()