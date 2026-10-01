import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation
from skimage import measure
from tqdm import tqdm

from equations import *


# ============================================================
# APS / REVTeX Standard Style Configuration
# ============================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIXGeneral'],
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'legend.fontsize': 6.5,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'lines.linewidth': 1.0,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'figure.figsize': (3.37, 5.2),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


# ============================================================
# Parameters / calculations for PANEL (a)
# ============================================================

N = 15
M = int(1e3)

use_3d = True
use_4d = True

profiles = []

for i in range(N):
    r, profile = get_bessel_profile(i + 1, N_points=M)
    profiles.append(profile)

profiles = np.array(profiles).T

W3, W4 = prepare_spatial_vdw_weights(r)

config_SI = {
    'N': N,
    'Gammas': np.ones(N) * 10,
    'tau': 1 / 5000,
    'power': 200e-6,
    'detuning': -3e6,
    'd': 10e-9
}

config = to_unitless(config_SI)

delta = -2
alpha_bif = upper_boundary(N, delta)

print(f'cusp: {cusp(N)}')
print(f'bifurcation: {alpha_bif}')

config_final = {
    'N': N,
    'gamma': config['gamma'],
    'mu': mu_spectrum(N),
    'tau': config['tau'],
    'alpha': 0.3,
    'delta': delta,
    'sigma': 30,
    'xi': np.ones(N),
    'nu': config['nu']
}

deltas = np.linspace(-5, 2, 300)
alphas = np.linspace(0, 1, 300)
                     
N_fixed_points = np.zeros((len(alphas), len(deltas)))
X_star = np.zeros((len(alphas), len(deltas)))

for i, d in tqdm(enumerate(deltas), total=len(deltas)):

    config_final['delta'] = d

    for j, a in enumerate(alphas):

        config_final['alpha'] = a

        fixed_points = fixed_points_num(
            system_numba_hidde,
            (config_final, profiles, W3, W4, use_3d, use_4d),
            num_tries=100
        )

        if len(fixed_points) > 0:

            X_stars = []

            for p in fixed_points:
                X_stars.append(np.sum(p[:N]))

            X_star[j, i] = np.max(X_stars)
            N_fixed_points[j, i] = len(fixed_points)

        else:
            X_star[j, i] = 0
            N_fixed_points[j, i] = 0


# ============================================================
# Boundary extraction
# ============================================================

is_multistable = N_fixed_points > 1

deltas_lower_clean, alphas_lower_clean = [], []
deltas_upper_clean, alphas_upper_clean = [], []

for i in range(len(deltas)):

    rows = np.where(is_multistable[:, i])[0]

    if len(rows) > 0:

        deltas_lower_clean.append(deltas[i])
        alphas_lower_clean.append(alphas[rows[0]])

        deltas_upper_clean.append(deltas[i])
        alphas_upper_clean.append(alphas[rows[-1]])

deltas_lower_clean = np.array(deltas_lower_clean)
alphas_lower_clean = np.array(alphas_lower_clean)

deltas_upper_clean = np.array(deltas_upper_clean)
alphas_upper_clean = np.array(alphas_upper_clean)

threshold = lasing_threshold(
    config_final,
    deltas,
    return_all=False
)


# ============================================================
# Function for PANEL (b)
# ============================================================

def fixed_point_relation(delta, alpha, x, N=15):
    return x * ((N * x + delta)**2 + 1) - alpha


# ------------------------------------------------------------
# Grid
# ------------------------------------------------------------

deltas_3d = np.linspace(-10, 2, 100)
alphas_3d = np.linspace(0, .6, 100)
xs = np.linspace(0, .65, 100)

D, A, X = np.meshgrid(
    deltas_3d,
    alphas_3d,
    xs,
    indexing='ij'
)

F = fixed_point_relation(D, A, X, N)


# ------------------------------------------------------------
# Extract F = 0 surface
# ------------------------------------------------------------

spacing = (
    deltas_3d[1] - deltas_3d[0],
    alphas_3d[1] - alphas_3d[0],
    xs[1] - xs[0]
)

verts, faces, normals, values = measure.marching_cubes(
    F,
    level=0,
    spacing=spacing,
    step_size=3
)


# ------------------------------------------------------------
# Restore physical coordinates
# ------------------------------------------------------------

verts[:, 0] += deltas_3d[0]
verts[:, 1] += alphas_3d[0]
verts[:, 2] += xs[0]


# ------------------------------------------------------------
# Triangulation
# ------------------------------------------------------------

tri = Triangulation(
    verts[:, 0],
    verts[:, 1],
    triangles=faces
)


# ============================================================
# ONE FIGURE — TWO STACKED PANELS
# ============================================================

fig = plt.figure(figsize=(3.37, 5.8))


# ============================================================
# PANEL (a): 2D Bifurcation Diagram
# ============================================================

ax1 = fig.add_subplot(2, 1, 1)

mesh = ax1.pcolormesh(
    deltas,
    alphas,
    X_star,
    shading='nearest',
    cmap='cividis',
    rasterized=True
)

cbar = fig.colorbar(
    mesh,
    ax=ax1,
    label=r'$X^\ast$',
    pad=0.02
)

# Analytical boundaries
ax1.plot(
    deltas,
    upper_boundary(N, deltas),
    c='k',
    label='0th Order Boundary',
    zorder=2
)

ax1.plot(
    deltas,
    lower_boundary(N, deltas),
    c='k',
    zorder=2
)


# Numerical multistability boundaries
ax1.plot(
    deltas_lower_clean,
    alphas_lower_clean,
    'k:',
    label='Num. Boundary',
    zorder=3,
    linewidth=0.7
)

ax1.plot(
    deltas_upper_clean,
    alphas_upper_clean,
    'k:',
    zorder=3,
    linewidth=0.7
)

# Lasing threshold
ax1.plot(
    deltas,
    threshold,
    color='r',
    linestyle='--',
    linewidth=0.7,
    label='Lasing Threshold',
    zorder=4
)



ax1.set_xlim(np.min(deltas), np.max(deltas))
ax1.set_ylim(np.min(alphas), np.max(alphas))

ax1.set_xlabel(r'$\delta$')
ax1.set_ylabel(r'$\alpha$')

ax1.set_title(
    '(a) Bifurcation Diagram',
    fontsize=9,
)

ax1.legend(
    loc='best',
    frameon=True,
    facecolor='white',
    edgecolor='none',
    fontsize=6
)


# ============================================================
# PANEL (b): 3D Equilibrium Surface
# ============================================================



ax2 = fig.add_subplot(
    2, 1, 2,
    projection='3d'
)

plot_bifurcation_boundaries = False

if plot_bifurcation_boundaries:
    # ============================================================
    # Compute 3D Bifurcation Boundary Lines
    # ============================================================

    # Domain bounded up to the cusp point
    delta_cusp = -np.sqrt(3)
    deltas_line = np.linspace(-8.0, delta_cusp, 100)

    # Guard against floating point underflow near delta = -sqrt(3)
    discriminant = np.maximum(0.0, N**2 * (deltas_line**2 - 3))

    # 1. Analytical fold values of x_bar (double roots)
    x_fold_low = (-2 * N * deltas_line - np.sqrt(discriminant)) / (3 * N**2)
    x_fold_high = (-2 * N * deltas_line + np.sqrt(discriminant)) / (3 * N**2)

    # 2. Corresponding alpha boundaries alpha_b^-(delta) and alpha_b^+(delta)
    alpha_b_minus = x_fold_low * ((N * x_fold_low + deltas_line)**2 + 1)
    alpha_b_plus = x_fold_high * ((N * x_fold_high + deltas_line)**2 + 1)

    # 3. Analytical outer x_bar values using sum of roots: x_outer = -2*delta/N - 2*x_fold
    x_outer_at_minus = - (2 * deltas_line / N) - 2 * x_fold_low
    x_outer_at_plus  = - (2 * deltas_line / N) - 2 * x_fold_high


    # ============================================================
    # Plot the 4 Boundary Lines on ax2 (PANEL b)
    # ============================================================

    # Line 1: Lower fold boundary line
    ax2.plot(deltas_line, alpha_b_minus, x_fold_low, color='k', linestyle='-', linewidth=1.2, zorder=10)

    # Line 2: Upper fold boundary line
    ax2.plot(deltas_line, alpha_b_plus, x_fold_high, color='k', linestyle='-', linewidth=1.2, zorder=10)

    # Line 3: Projection onto upper branch at lower boundary
    ax2.plot(deltas_line, alpha_b_minus, x_outer_at_minus, color='k', linestyle='--', linewidth=1.0, zorder=10)

    # Line 4: Projection onto lower branch at upper boundary
    ax2.plot(deltas_line, alpha_b_plus, x_outer_at_plus, color='k', linestyle='--', linewidth=1.0, zorder=10)

    # Optional: Highlight the Cusp Point where all 4 lines meet
    x_cusp = -delta_cusp / (3 * N)
    alpha_cusp = x_cusp * ((N * x_cusp + delta_cusp) ** 2 + 1)
    ax2.scatter(
        [delta_cusp],
        [alpha_cusp],
        [x_cusp],
        color='red',
        s=15,
        zorder=100,
        label='Cusp Point',
    )

surf = ax2.plot_trisurf(
    tri,
    verts[:, 2],
    cmap='cividis',
    linewidth=0,
    edgecolor='none',
    antialiased=False
)

surf.set_rasterized(True)

ax2.set_xlabel(r'$\delta$', labelpad=2)
ax2.set_ylabel(r'$\alpha$', labelpad=2)
ax2.set_zlabel('')
ax2.text2D(
    -0.13, 0.5,
    r'$\overline{x}$',
    transform=ax2.transAxes,
    rotation=90,
    va='center',
    ha='center'
)
ax2.set_xlim(np.min(deltas_3d), np.max(deltas_3d))
ax2.set_ylim(np.min(alphas_3d), np.max(alphas_3d))
ax2.set_zlim(np.min(xs), np.max(xs))

ax2.view_init(
    elev=25,
    azim=-125
)

ax2.set_title(
    '(b) 3D Bifurcation Surface',
    fontsize=9
)


# ============================================================
# Final layout
# ============================================================

plt.subplots_adjust(
    left=0.12,
    right=0.95,
    bottom=0.07,
    top=0.96,
    hspace=0.35
)

#plt.tight_layout()
plt.savefig(
    'fig1.pdf',
    dpi=300,
    bbox_inches='tight'
)

plt.show()