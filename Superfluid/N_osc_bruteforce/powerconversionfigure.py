import matplotlib.pyplot as plt

# matplotlib.colors.LogNorm is imported for logarithmic color scaling
from matplotlib.colors import LogNorm
import numpy as np

# -----------------------------------------------------------------------------
# 1. System Parameters
# -----------------------------------------------------------------------------
rho = 145  # kg/m3
rho_s = 145  # kg/m^3
a_vdw = 2.6e-24  # m^5/s^2
kappa = 11.45e6  # Hz, FWHM
G = 2e16  # Hz/m
R = 3e-3  # m
beta = 1.5e6
f = 2.818e14  # Hz
kappa_ex = None  # Hz, FWHM
verbose = True

if kappa_ex is None:
    kappa_ex = kappa / 2  # assumes critical coupling

kappa_rad_HWHM = np.pi * kappa  # rad/s, HWHM
kappa_ex_rad_HWHM = np.pi * kappa_ex
G_rad = 2 * np.pi * G
omega = 2 * np.pi * f

C = (
    2
    * beta
    * kappa_ex_rad_HWHM
    * G_rad**2
    / (3 * np.pi * a_vdw * rho * omega * kappa_rad_HWHM**3 * R**2)
)
k = 1 / (kappa_rad_HWHM / G_rad) / 1e9  # Proportionality constant for sigma

# -----------------------------------------------------------------------------
# 2. Grid Generation (d and alpha)
# -----------------------------------------------------------------------------
d_min_nm, d_max_nm = 5, 25  # Range for d in nm
alpha_min, alpha_max = 0.1, 1.0  # Range for alpha

d_nm_vec = np.geomspace(d_min_nm, d_max_nm, 500)
alpha_vec = np.geomspace(alpha_min, alpha_max, 500)

# Create 2D meshgrid
D_nm, ALPHA = np.meshgrid(d_nm_vec, alpha_vec)
D_m = D_nm * 1e-9  # Convert d from nm to meters for calculation

# Solve for P (in microwatts): P = alpha / (C * d^4)
P_watts = ALPHA / (C * (D_m**4))
P_uW = P_watts / 1e-6


# -----------------------------------------------------------------------------
# 3. Secondary Axis Functions
# -----------------------------------------------------------------------------
def d_to_sigma(d):
    return k * d


def sigma_to_d(sigma):
    return sigma / k


# -----------------------------------------------------------------------------
# 4. Heatmap Plotting with Colorbar & Overlay Contours
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))

# Plot continuous heatmap with logarithmic color scale
heatmap = ax.pcolormesh(
    D_nm, ALPHA, P_uW, norm=LogNorm(), cmap="viridis", shading="auto"
)

# Add colorbar
cbar = fig.colorbar(heatmap, ax=ax, pad=0.02)
cbar.set_label(r"Power $P$ [$\mu\mathrm{W}$]", fontsize=12)

# Optional: Add white contour lines for specific power values for easy reading
P_levels_contour = [1, 5, 10, 50, 100, 500, 1000]
contours = ax.contour(
    D_nm,
    ALPHA,
    P_uW,
    levels=P_levels_contour,
    colors="white",
    linewidths=0.8,
    alpha=0.6,
)
ax.clabel(contours, inline=True, fontsize=8, fmt="%g μW")

# Set Primary Axes
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("d [nm]", fontsize=12)
ax.set_ylabel(r"$\alpha$", fontsize=12)
ax.set_ylim(alpha_min, alpha_max)
ax.set_title(
    r"Conversion Map: Power $P$ as function of $d$ and $\alpha$",
    fontsize=13,
    pad=25,
)

# Add Secondary X-Axis at the Top for sigma
secax = ax.secondary_xaxis("top", functions=(d_to_sigma, sigma_to_d))
secax.set_xlabel(r"$\sigma$", fontsize=12, labelpad=8)

plt.tight_layout()
plt.show()