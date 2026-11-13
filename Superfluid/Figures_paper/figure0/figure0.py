
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.special import j0, jn_zeros

# ------------------------------------------------------------
# Plot styling
# ------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "legend.fontsize": 6.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
R = 1.0
d = 0.1

# Bessel mode amplitudes and mode indices
# Reduced amplitudes so that d is much larger than eta
amplitudes = [0.07, 0.05, -0.04]
mode_numbers = [2, 5, 7]

# Substrate slab
substrate_top = 0.0
substrate_bottom = -0.03

# Optical mode: vertical band at the right edge
optical_band_width = 0.045 * R

# Radial coordinate
r = np.linspace(0, R, 1800)

# ------------------------------------------------------------
# Bessel modes with Neumann boundary conditions
#
# q_n(r) = A_n J_0(alpha_n r/R)
# q_n'(R) = 0 when J_1(alpha_n) = 0
# ------------------------------------------------------------
roots = jn_zeros(1, max(mode_numbers))

modes = np.array([
    A * j0(roots[n - 1] * r / R)
    for A, n in zip(amplitudes, mode_numbers)
])

# Total film displacement and thickness
eta = np.sum(modes, axis=0)
h = d + eta

# ------------------------------------------------------------
# Create figure
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.37, 2.5))

substrate_color = "0.76"

# Vertical plot limits
ymin = substrate_bottom
ymax = max(h.max(), d)+0.01

# ------------------------------------------------------------
# Substrate: single continuous slab
# ------------------------------------------------------------
ax.add_patch(Rectangle(
    (0, substrate_bottom),
    R,
    substrate_top - substrate_bottom,
    facecolor=substrate_color,
    edgecolor="none",
    linewidth=0.9,
    zorder=0
))

ax.text(
    0.52 * R,
    0.5 * (substrate_bottom + substrate_top),
    "Substrate",
    ha="center",
    va="center",
    fontsize=8.5,
    color="0.25",
    zorder=2
)


# ------------------------------------------------------------
# Optical mode: continuous, horizontally symmetric alpha gradient
# ------------------------------------------------------------
optical_band_width = 0.01 * R

nx, ny = 500, 70
x_opt = np.linspace(R - optical_band_width, R + optical_band_width, nx)
y_opt = np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x_opt, y_opt)

# Gaussian opacity, symmetric about x = R
sigma_x = optical_band_width / 3
alpha_max = 1
alpha_gradient = alpha_max * np.exp(
    -0.5 * ((X - R) / sigma_x) ** 2
)

# Constant hue with continuously varying transparency
rgba = np.zeros((ny, nx, 4))
rgb = plt.matplotlib.colors.to_rgba("C3")
rgba[..., :3] = rgb[:3]
rgba[..., 3] = alpha_gradient

ax.imshow(
    rgba,
    origin="lower",
    extent=(x_opt.min(), x_opt.max(), ymin, ymax),
    aspect="auto",
    interpolation="bilinear",
    zorder=100
)

# ------------------------------------------------------------
# Film surface and mode profiles
# ------------------------------------------------------------
ax.fill_between(
    r, 0, h,
    color="C0",
    alpha=0.10,
    zorder=0,
    edgecolor="none"
)

# Total film thickness h(r)
ax.plot(
    r, h,
    color="black",
    linewidth=1.25,
    label=r"Total film thickness $h(r)$",
    zorder=6
)

# Mean film thickness d
ax.plot(
    [0, R], [d, d],
    color="0.25",
    linestyle=(0, (7, 5)),
    linewidth=0.85,
    zorder=5
)

# Individual modes: same colour, alpha = 0.6
for i, q in enumerate(modes):
    ax.plot(
        r, d + q,
        color="C1",
        linewidth=0.9,
        alpha=0.6,
        label="Individual modes" if i == 0 else None,
        zorder=5
    )

# ------------------------------------------------------------
# Helper: vertical dimension arrow with a zigzag scale break
# ------------------------------------------------------------
def broken_dimension(
    x, y0, y1, label,
    text_dx=0.018,
    text_side="right",
    color="black",
    fontsize=9,
    break_fraction=0.5,
    text_y_frac=0.5
):
    """
    Draw a vertical double-headed dimension arrow with a zigzag
    break. The zigzag indicates that the dimension is not drawn
    to scale.
    """
    if y1 < y0:
        y0, y1 = y1, y0

    break_y = y0 + break_fraction * (y1 - y0)
    gap = 0.01

    # Arrow segments below and above the zigzag
    ax.annotate(
        "",
        xy=(x, 0.97*(break_y - gap / 2)),
        xytext=(x, y0),
        arrowprops=dict(
            arrowstyle="<|-",
            color=color,
            linewidth=0.9,
            shrinkA=0,
            shrinkB=0
        ),
        zorder=9
    )

    ax.annotate(
        "",
        xy=(x, y1),
        xytext=(x, 1.03*(break_y + gap / 2)),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            linewidth=0.9,
            shrinkA=0,
            shrinkB=0
        ),
        zorder=9
    )

    # Zigzag scale-break symbol
    size = 0.004
    ax.plot(
        [
            x - size, x + size,
            x - size, x + size,
            x - size
        ],
        [
            break_y + 2 * size,
            break_y + size,
            break_y,
            break_y - size,
            break_y - 2 * size
        ],
        color=color,
        linewidth=0.9,
        solid_capstyle="round",
        zorder=10,
        clip_on=False
    )

    # Dimension label
    if text_side == "right":
        tx, ha = x + text_dx, "left"
    else:
        tx, ha = x - text_dx, "right"

    ax.text(
        tx,
        text_y_frac * (y0 + y1),
        label,
        ha=ha,
        va="center",
        fontsize=fontsize,
        color=color,
        zorder=11
    )

# ------------------------------------------------------------
# Mean film thickness d, with scale break
# ------------------------------------------------------------
x_d = 0.0

broken_dimension(
    x=x_d,
    y0=0,
    y1=d,
    label=r"$d$",
    text_dx=0.01,
    text_side="left",
    fontsize=9,
    break_fraction=0.3,
    text_y_frac=0.6
)

# ------------------------------------------------------------
# Total film thickness h(r), with scale break
# ------------------------------------------------------------
x_h = 0.035 * R
h_at_x = np.interp(x_h, r, h)

broken_dimension(
    x=x_h,
    y0=0,
    y1=h_at_x,
    label=r"$h(r)$",
    text_dx=0.01,
    text_side="right",
    fontsize=8.5,
    break_fraction=0.3*d/h_at_x,
    text_y_frac=0.3
)

# ------------------------------------------------------------
# Total displacement eta(r) = h(r) - d
# ------------------------------------------------------------
x_eta = 0
eta_at_x = np.interp(x_eta, r, eta)
h_eta = d + eta_at_x

ax.annotate(
    "",
    xy=(x_eta, 1.02*h_eta),
    xytext=(x_eta, 0.98*d),
    arrowprops=dict(
        arrowstyle="<|-|>",
        color="black",
        linewidth=0.9
    ),
    zorder=9
)

ax.text(
    x_eta - 0.01 * R,
    0.5 * (h_eta + d),
    r"$\eta(r)$",
    ha="right",
    va="center",
    fontsize=8.5,
    color="black",
    zorder=10
)

# ------------------------------------------------------------
# Individual mode displacement eta_i(r)
# ------------------------------------------------------------
mode_to_label = 0
x_mode = 0.1 * R
q_at_x = np.interp(x_mode, r, modes[mode_to_label])

ax.annotate(
    "",
    xy=(x_mode, 1.02*(d + q_at_x)),
    xytext=(x_mode, 0.98*d),
    arrowprops=dict(
        arrowstyle="<|-|>",
        color="C1",
        linewidth=0.9
    ),
    zorder=9
)

ax.text(
    x_mode + 0.012 * R,
    d + q_at_x / 2,
    r"$\eta_i(r)$",
    ha="left",
    va="center",
    fontsize=8.5,
    color="C1",
    zorder=10
)

# ------------------------------------------------------------
# Horizontal coordinate: show only 0 and R
# ------------------------------------------------------------
# Regular 0-to-R axis with end ticks (no arrows)
y_span = substrate_bottom - 0.02

ax.plot([0, R], [y_span, y_span],
        color="black", lw=1.2, clip_on=False, zorder=5)

tick_height = 0.012 * R
ax.plot([0, 0], [y_span - tick_height/2, y_span + tick_height/2],
        color="black", lw=1.2, clip_on=False, zorder=5)
ax.plot([R, R], [y_span - tick_height/2, y_span + tick_height/2],
        color="black", lw=1.2, clip_on=False, zorder=5)

# Endpoint labels
ax.text(0, y_span - 1.5*tick_height, r"$0$",
        ha="center", va="top")
ax.text(R, y_span - 1.5*tick_height, r"$R$",
        ha="center", va="top")

# ------------------------------------------------------------
# Styling and legend
# ------------------------------------------------------------
ax.set_xlim(-0.07 * R, R + optical_band_width)
ax.set_ylim(ymin, ymax)
ax.axis("off")

handles, labels = ax.get_legend_handles_labels()
unique = {}

for handle, label in zip(handles, labels):
    if label and label not in unique:
        unique[label] = handle

ax.legend(
    unique.values(),
    unique.keys(),
    loc="upper left",
    bbox_to_anchor=(0.25, 0.35),
    frameon=False,
    fontsize=6.5,
    borderaxespad=0
)

fig.tight_layout()

# ------------------------------------------------------------
# Save and display
# ------------------------------------------------------------
fig.savefig(
    "resonator_geometry_slab_optical_mode.png",
    dpi=300,
    bbox_inches="tight",
    transparent=True
)

fig.savefig(
    "resonator_geometry_slab_optical_mode.pdf",
    bbox_inches="tight",
    transparent=True
)

plt.show()
