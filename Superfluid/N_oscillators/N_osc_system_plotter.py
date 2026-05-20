import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from matplotlib.patches import Rectangle
from scipy.integrate import solve_ivp
from matplotlib.gridspec import GridSpec

from N_osc_eqs import *

verbose = True
np.set_printoptions(precision=4)

# -----------------------------
# parameter ranges
# -----------------------------
delta_min, delta_max = -5, 2
deltas = np.linspace(delta_min, delta_max, 100)
0
alpha_min, alpha_max = 0, np.abs(delta_min)
alphas = np.linspace(alpha_min, alpha_max, 100)

N_min, N_max = 2, 25

# -----------------------------
# initial parameters
# -----------------------------
alpha0, delta0 = 0.3, 0.0
tau0 = 1.0
N0 = 2

gamma = 0.05
mus = mu_spectrum(N0)
#
print(mus)
gammas = np.full(N0, gamma)

#threshold_polys = []
#for N in range(N_min, N_max+1):
#    threshold_polys.append(derive_threshold_polynomials(mu_spectrum(N), np.full(N, gamma), tau0))

# ICs
x0 = np.zeros(N0)
y0 = np.zeros(N0)
z0 = 0.0
T0 = 100

colors = ['r', 'b', 'g', 'm']

# -----------------------------
# FIGURE
# -----------------------------
fig = plt.figure(figsize=(16, 10))
fig.subplots_adjust(bottom=0.32)
gs = GridSpec(2, 2, hspace=0.4, wspace=0.3)

# LEFT COLUMN
ax1 = fig.add_subplot(gs[0, 0])  # bifurcation
ax2 = fig.add_subplot(gs[1, 0])  # eigenvalues

# CENTER COLUMN
ax3 = fig.add_subplot(gs[0, 1])  # timetrace
ax4 = fig.add_subplot(gs[1, 1])  # open

# =============================
# BIFURCATION
# =============================
ax1.fill_between(deltas, alpha_min, alpha_max, color='lightblue', alpha=0.3)

bif_lower, = ax1.plot([], [], 'k', lw=2)
bif_upper, = ax1.plot([], [], 'k', lw=2)

lasing_lines = []
for i in range(40):
    lasing_line, = ax1.plot([], [], 'r', lw=1)
    lasing_lines.append(lasing_line)

point, = ax1.plot([], [], 'ko')

ax1.set_xlim(delta_min, delta_max)
ax1.set_ylim(alpha_min, N0*np.abs(delta_min))
ax1.set_xlabel(r'$\delta$')
ax1.set_ylabel(r'$\alpha$')
ax1.set_title("Bifurcation diagram")
ax1.grid()

# =============================
# EIGENVALUES
# =============================
scatters = [ax2.scatter([], [], color=c, s=10) for c in ['r', 'g', 'b']]

ax2.axhline(0, color='gray')
ax2.axvline(0, color='gray')
ax2.set_xlim(-0.03, 1)
ax2.set_ylim(0, 18)
ax2.set_title("Eigenvalues")

# =============================
# PHASE SPACE
# =============================
traj1, = ax3.plot([], [], 'k')

ax3.set_title("(X, T)")
ax4.set_title("()")

# =============================
# TIME SERIES
# =============================


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

sA = make_slider(0.5, 0.23, 'α', alpha_min, alpha_max, alpha0)
sD = make_slider(0.75, 0.23, 'δ', delta_min, delta_max, delta0)
sT = make_slider(0.5, 0.19, 'τ', 0.1, 5, tau0)

sN = make_slider(0.75, 0.19, 'N', N_min, N_max, N0, step=1)
sTime = make_slider(0.75, 0.07, 'T', 1, 500, T0)

# =============================
# UPDATE
# =============================
def update(val):
    if verbose: print('____________________________')

    alpha, delta = sA.val, sD.val
    tau = sT.val
    N = int(sN.val)

    T = sTime.val

    mus = mu_spectrum(N)
    #ax1.set_ylim(alpha_min, 2*abs(delta_min)*(abs(delta_min)**2-3)/N/27)
    ax1.set_ylim(alpha_min, alpha_max)
    #mu_spectrum(N0)
    gammas = np.full(N, gamma)

    # bifurcation
    bif_lower.set_data(deltas, lower_boundary(N, deltas))
    bif_upper.set_data(deltas, upper_boundary(N, deltas))

    
    thresholds = lasing_threshold2(N, deltas, tau, mus, gammas)

    for lasing_line in lasing_lines:
        lasing_line.set_data([], [])

    for i, lasing_line in enumerate(lasing_lines):
        if i < len(thresholds):
            lasing_line.set_data(deltas, thresholds[i])
        else:
            lasing_line.set_data([], [])

    point.set_data([delta], [alpha])

    # eigenvalues
    roots, eigvals, eigvecs = compute_eigs(N, mus, alpha, delta, tau, gammas)


    for i_root, (vals, vecs, scatter) in enumerate(zip(eigvals, eigvecs, scatters)):

        scatter.set_offsets(np.c_[vals.real, vals.imag])



    # solve system
    y0 = np.zeros(2*N+1)
    t_eval = np.linspace(0, T, 5000)
    sol = solve_ivp(
        lambda time,X: system(time, X, alpha, delta, mus, gammas, tau),
        (0, T), y0, t_eval=t_eval
    )
    #print(np.shape(sol.y))
    X = np.sum(sol.y[:-1:2,:], axis=0)
    #print(np.shape(X))
    # trajectories
    traj1.set_data(t_eval, X)
    ax3.relim()
    ax3.autoscale_view()

    fig.canvas.draw_idle()

# connect sliders
for s in [sA,sD,sT,sN,sTime]:
    s.on_changed(update)


update(None)
plt.show()