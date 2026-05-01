import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from matplotlib.patches import Rectangle
from scipy.integrate import solve_ivp
from matplotlib.gridspec import GridSpec

import warnings
warnings.filterwarnings('ignore')

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
T0 = 499

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
ax3 = fig.add_subplot(gs[0, 1])  # (x1,v1,z)
ax4 = fig.add_subplot(gs[1, 1])  # (x2,v2,z)
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
traj1, = ax3.plot([], [], 'k')
traj2, = ax4.plot([], [], 'k')

traj12, = ax5.plot([], [], 'k')

ax3.set_title("projection limit cycle 1")
ax4.set_title("projection limit cycle 2")
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
box = Rectangle((0.45, 0.02), 0.5, 0.25,
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

sA = make_slider(0.5, 0.23, 'α', 0, 5, alpha0)
sD = make_slider(0.75, 0.23, 'δ', delta_min, delta_max, delta0)
sT = make_slider(0.5, 0.07, 'τ', 0.1, 5, tau0)
sA2 = make_slider(0.5, 0.19, 'α2', 0, 5, alpha0)

srho = make_slider(0.75, 0.19, r'$\rho$', 0.1, 3, rho0, step=.1)

sG1 = make_slider(0.75, 0.15, 'γ1', 0, 2, gamma10)
sG2 = make_slider(0.5, 0.15, 'γ2', 0, 2, gamma20)

sX10 = make_slider(0.75, 0.11, 'x1₀', -5, 5, x10, step=.1)
sX20 = make_slider(0.5, 0.11, 'x2₀', -5, 5, x20, step=.1)

sTime = make_slider(0.75, 0.07, 'T', 1, 500, T0)

# =============================
# UPDATE
# =============================
def update(val):

    alpha, delta = sA.val, sD.val
    alpha2 = sA2.val
    tau = sT.val
    rho = srho.val
    g1, g2 = sG1.val, sG2.val

    T = sTime.val

    # bifurcation
    bif_lower.set_data(deltas, lower_boundary(deltas, rho)*sigma(rho))
    bif_upper.set_data(deltas, upper_boundary(deltas, rho)*sigma(rho))

    #thresholds = lasing_threshold(deltas, tau, rho, g1, g2)

    #for threshold, lasing_line in zip(thresholds, lasing_lines):
        #lasing_line.set_data(deltas, threshold*sigma(rho))

    point.set_data([delta], [alpha*sigma(rho)])

    # eigenvalues
    roots, eigvals, eigvecs = compute_eigs(alpha, delta, tau, rho, g1, g2)

    if not hasattr(update, "eig_quivers"):
        update.eig_quivers = []
    else:
        for q in update.eig_quivers:
            q.remove()
        update.eig_quivers = []

    
    for i in range(4):
        if i < len(eigvals):
            eigs = eigvals[i]
            scatters[i].set_offsets(np.c_[eigs.real, eigs.imag])
        else:
            scatters[i].set_offsets(np.empty((0,2)))

    # solve system
    y0 = [sX10.val, 0, sX20.val, 0, z0]

    t_eval1 = np.linspace(0, T, 5000)

    sol1 = solve_ivp(
        lambda t,y: system(t, y, alpha, delta, tau, rho, g1, g2),
        (0, T), y0, t_eval=t_eval1
    )

    yT = sol1.y[:, -1]  # final state of first simulation

    t_eval2 = np.linspace(T, 500, 5000)

    sol2 = solve_ivp(
        lambda t, y: system(t, y, alpha2, delta, tau, rho, g1, g2),
        (T, 500),
        yT,
        t_eval=t_eval2
    )

    # --- combine results (avoid duplicating time T) ---
    t_eval = np.concatenate((sol1.t, sol2.t[1:]))
    y = np.hstack((sol1.y, sol2.y[:, 1:]))

    # unpack
    x1, v1, x2, v2, z = y

    traj12.set_data(x1, x2)


    ax5.set_xlim(x1.min(), x1.max())
    ax5.set_ylim(x2.min(), x2.max())

    # time series
    line_x1.set_data(t_eval, x1)
    line_v1.set_data(t_eval, v1)
    line_z1.set_data(t_eval, z)

    line_x2.set_data(t_eval, x2)
    line_v2.set_data(t_eval, v2)

    for ax in [ax6, ax7]:
        ax.set_xlim(0, 500)

    ax6.set_ylim(min(x1.min(), v1.min(), z.min()),
                 max(x1.max(), v1.max(), z.max()))

    ax7.set_ylim(min(x2.min(), v2.min()),
                 max(x2.max(), v2.max()))
    

    scale = 0.15
    lreal1 = None
    limag1 = None
    lreal2 = None
    limag2 = None

    vreal1 = None
    vimag1 = None
    vreal2 = None
    vimag2 = None
    for i, root in enumerate(roots):
        vals = eigvals[i]
        vecs = eigvecs[i]
        print(vals, vecs)
        for j in range(len(vals)):
            l = vals[j]
            v = vecs[:, j]
            print(l, v)
            # test whether eigenvector has an imaginary part
            if (np.abs(np.imag(v)[0])>1e-9 or np.abs(np.imag(v)[1])>1e-9 or np.abs(np.imag(v)[2])>1e-9 or np.abs(np.imag(v)[3])>1e-9 or np.abs(np.imag(v)[4])>1e-9):
                print('IMAGINARY')
                vreal = np.real(v)
                print(vreal1, vreal)

                if vreal1 is None:
                    vreal1 = vreal
                    vimag1 = np.imag(v)
                    print('1', vreal1, vimag1)
                    print(rotate(vreal1), rotate(vimag1))
                    projection = project_onto_plane(y, vreal1, vimag1)

                    traj1.set_data(projection[0], projection[1])
                    ax3.set_xlim(np.min(projection[0]), np.max(projection[0]))
                    ax3.set_ylim(np.min(projection[1]), np.max(projection[1]))

                elif (np.abs(vreal[0] - vreal1[0]) > 1e-9 or np.abs(vreal[1] - vreal1[1]) > 1e-9 or np.abs(vreal[2] - vreal1[2]) > 1e-9) and vreal2 is None:
                    vreal2 = vreal
                    vimag2 = np.imag(v)
                    print('2', vreal2, vimag2)
                    projection = project_onto_plane(y, vreal2, vimag2)

                    traj2.set_data(projection[0], projection[1])
                    ax4.set_xlim(np.min(projection[0]), np.max(projection[0]))
                    ax4.set_ylim(np.min(projection[1]), np.max(projection[1]))

            else:
                print('NOT IMAGINARY')
    
    print('____________________________________________________________________________')

    fig.canvas.draw_idle()

# connect sliders
for s in [sA,sA2,sD,sT,srho,sG1,sG2,sX10,sX20,sTime]:
    s.on_changed(update)

update(None)
plt.show()