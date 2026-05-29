import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, CheckButtons
from matplotlib.patches import Rectangle
from scipy.integrate import solve_ivp
from matplotlib.gridspec import GridSpec

from N_osc_eqs_w_HO import *

# ============================================================
# SETTINGS
# ============================================================
verbose = False
np.set_printoptions(precision=4)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
})

gamma = 0.05
colors = ['r', 'b', 'g', 'm']

# ============================================================
# PARAMETER GRIDS (BIFURCATION MAP)
# ============================================================
delta_min, delta_max = -5, 2
alpha_min, alpha_max = 0, 1

N_delta = 100
N_alpha = 100

deltas = np.linspace(delta_min, delta_max, N_delta)
deltas_eff = np.linspace(0, delta_max, N_delta)
alphas = np.linspace(alpha_min, alpha_max, N_alpha)

D, A = np.meshgrid(deltas, alphas, indexing='ij')
Z = np.zeros((N_delta, N_alpha))

D_eff, A = np.meshgrid(deltas_eff, alphas, indexing='ij')
Z_eff = np.zeros((N_delta, N_alpha))

N_min, N_max = 1, 20

# ============================================================
# INITIAL PARAMETERS
# ============================================================
N0 = 2
T0 = 100

params = {
        'N': N0,
        'sigma': 20,
        'tau': 1.0,
        'alpha': 0,
        'delta': 0,
        'gamma': np.full(N0, 0.05),
        'mu': mu_spectrum(N0),
        'chi_ijk': np.load('./tensors/chi_ijk.npy'),
    }

# ============================================================
# INITIAL CONDITIONS
# ============================================================
x0 = np.zeros(N0)
y0 = np.zeros(N0)
z0 = 0.0

# ============================================================
# FIGURE LAYOUT
# ============================================================
fig = plt.figure(figsize=(16, 10))
fig.subplots_adjust(bottom=0.32)

gs = GridSpec(1, 2, hspace=0.4, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])  # bifurcation
ax2 = fig.add_subplot(gs[0, 1])  # eigenvalues

# ============================================================
# ============================================================
# 1. BIFURCATION PANEL (ax1)
# ============================================================

# curves
bif_lower, = ax1.plot([], [], 'k', lw=2)
bif_upper, = ax1.plot([], [], 'k', lw=2)

lasing_lines = [ax1.plot([], [], 'r', lw=1)[0] for _ in range(4 * N_max - 2)]
point, = ax1.plot([], [], 'ko')

ax1.set_xlim(delta_min, delta_max)
ax1.set_ylim(alpha_min, alpha_max)
ax1.set_xlabel(r'$\delta$')
ax1.set_ylabel(r'$\alpha$')
ax1.set_title("Bifurcation diagram")
ax1.grid()

# ============================================================
# 2. FIXED POINTS PANEL (ax2)
# ============================================================
points_num = [ax2.plot([], [], 'ko', markersize=4)[0] for _ in range(10)]
points_0th_order = [ax2.plot([], [], 'ro', markersize=4)[0] for _ in range(10)]
points_1st_order = [ax2.plot([], [], 'bo', markersize=4)[0] for _ in range(10)]

ax2.set_ylabel(r'$x_i^*$')
ax2.set_xlabel(r'$z^*$')
ax2.set_xlim(0, 0.1*params['sigma'])
ax2.set_ylim(0, 0.1*params['sigma'])
ax2.set_title("Lasing thresholds")

# ============================================================
# SLIDERS
# ============================================================
box = Rectangle((0.05, 0.05), 0.9, 0.1,
                transform=fig.transFigure,
                fill=False, linewidth=2)
fig.patches.append(box)


def make_slider_with_box(x, y, label, vmin, vmax, vinit, step=None):
    # slider axis
    ax_slider = fig.add_axes([x, y, 0.15, 0.02])
    slider = Slider(ax_slider, label, vmin, vmax, valinit=vinit, valstep=step)

    # textbox axis (to the right of slider)
    ax_box = fig.add_axes([x + 0.154, y - 0.002, 0.06, 0.03])
    textbox = TextBox(ax_box, "", initial=str(vinit))

    # sync: slider → box
    def slider_update(val):
        textbox.set_val(f"{val:.4g}")

    slider.on_changed(slider_update)

    # sync: box → slider
    def box_submit(text):
        try:
            val = float(text)
            if vmin <= val <= vmax:
                slider.set_val(val)
        except ValueError:
            pass  # ignore invalid input

    textbox.on_submit(box_submit)

    return slider, textbox


sA, boxA = make_slider_with_box(0.1, 0.11, 'α', alpha_min, alpha_max, params['alpha'])
sD, boxD = make_slider_with_box(0.1, 0.07, r'$\delta$', delta_min, delta_max, params['delta'])
sT, boxT = make_slider_with_box(0.4, 0.11, 'τ', 0.1, 5, params['tau'])
sN, boxN = make_slider_with_box(0.4, 0.07, 'N', N_min, N_max, N0, step=1)
sS, boxS = make_slider_with_box(0.7, 0.11, r'$\sigma$', 10, 40, params['sigma'], step=.5)
sTime, boxTime = make_slider_with_box(0.7, 0.07, 'T', 1, 500, T0)


# ============================================================
# UPDATE FUNCTION
# ============================================================
def update(val):#, update_cmap=False, update_cmap_eff=True, update_solver=False, update_thresholds=False):

    if verbose: print("Updating...")

    params['alpha'] = sA.val
    params['delta'] = sD.val
    params['tau'] = sT.val
    params['sigma'] = sS.val
    params['N'] = int(sN.val)
    params['mu'] = mu_spectrum(params['N'])
    params['gamma'] = np.full(params['N'], gamma)

    T = sTime.val

    bif_lower.set_data(deltas, lower_boundary(params['N'], deltas))
    bif_upper.set_data(deltas, upper_boundary(params['N'], deltas))
        
    point.set_data([params['delta']], [params['alpha']])
    
    ps_0th_order = fixed_points_0th_order(params)
    print(f'ps_0th_order:{ps_0th_order}')
    for i, plot in enumerate(points_0th_order):
        if i < len(ps_0th_order):
            p = ps_0th_order[i]
            plot.set_data([p], [p])
        else:
            plot.set_data([], [])

    ps_1st_order = fixed_points_1st_order(params)
    print(f'ps_1st_order:{ps_1st_order}')
    for i, plot in enumerate(points_1st_order):
        if i < len(ps_1st_order):
            p = ps_1st_order[i]
            x = p[:int(params['N'])]
            print(x, np.full_like(x, p[-1]))
            plot.set_data(x, np.full_like(x, p[-1]))
        else:
            plot.set_data([], [])


    ps_num = fixed_points_num(params)
    print(f'ps_num:{ps_num}')
    for i, plot in enumerate(points_num):
        if i < len(ps_num):
            p = ps_num[i]
            x = p[:int(params['N'])]
            print(x, np.full_like(x, p[-1]))
            plot.set_data(x, np.full_like(x, p[-1]))
        else:
            plot.set_data([], [])



    
        

    print('_________________________')
    fig.canvas.draw_idle()


# ============================================================
# CONNECT SLIDERS
# ============================================================
'''sA.on_changed(lambda val: update(val,
                                 update_solver=True,))
sD.on_changed(lambda val: update(val,
                                 update_solver=True))
sN.on_changed(lambda val: update(val,
                                 update_cmap=True,
                                 update_solver=True,
                                 update_thresholds=True))
sT.on_changed(lambda val: update(val,
                                 update_solver=True,
                                 update_thresholds=True))
sTime.on_changed(lambda val: update(val,
                                 update_solver=True))'''

for s in [sA, sD, sN, sT, sTime, sS]:
    s.on_changed(update)


# initial draw
update(None)#, update_cmap=True, update_cmap_eff=True, update_solver=True, update_thresholds=True)

plt.show()