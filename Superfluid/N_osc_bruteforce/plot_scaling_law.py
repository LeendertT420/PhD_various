import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. PARAMETERS (Under-the-hood geometry)
# ==========================================
gamma = 0.75
A = 1.0
x_c = 2.0
T_total = 10

# ==========================================
# 2. GENERATE DATA
# ==========================================

# --- Linear Scale Data ---
x_left = np.linspace(x_c - 1.5, x_c - 1e-4, 500)
x_right = np.linspace(x_c + 1e-4, x_c + 1.5, 500)
x_lin = np.concatenate([x_left, [np.nan], x_right])

T_ideal_lin = A * (x_lin - x_c) ** (-gamma)
T_trunc_lin = np.minimum(T_ideal_lin, T_total)

# --- Log-Log Scale Data (vs distance |x - x_c|) ---
dx_log = np.logspace(-3, 1, 1000)
T_ideal_log = A * (dx_log ** (-gamma))
T_trunc_log = np.minimum(T_ideal_log, T_total)

# ==========================================
# 3. PLOTTING
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
plt.subplots_adjust(wspace=0.28)

# ------------------------------------------
# SUBPLOT 1: LINEAR SCALE
# ------------------------------------------
ax1.plot(
    x_lin,
    T_ideal_lin,
    "k--",
    linewidth=1.5,
    label=r"Ideal $\overline{T}(x) \sim |x - x_c|^{-\gamma}$",
)
ax1.plot(
    x_lin,
    T_trunc_lin,
    color="crimson",
    linewidth=2.2,
    label=r"Truncated $\overline{T}(x)$",
)

# Reference lines
ax1.axvline(
    x=x_c,
    color="navy",
    linestyle=":",
    linewidth=2
)
ax1.axhline(
    y=T_total,
    color="gray",
    linestyle="-.",
    linewidth=1.2
)

ax1.set_ylim(0, T_total * 1.25)
ax1.set_xlabel("$x$", fontsize=12)
ax1.set_ylabel(r"$\overline{T}(x)$", fontsize=12)
ax1.set_title("Linear Scale", fontsize=13, fontweight="bold")

# Replace numerical ticks with symbolic position markers
ax1.set_xticks([x_c])
ax1.set_xticklabels([r"$x_c$"], fontsize=12)
ax1.set_yticks([T_total])
ax1.set_yticklabels([r"$T_{\text{total}}$"], fontsize=12)

ax1.grid(True, linestyle=":", alpha=0.5)
ax1.legend(loc="upper right", fontsize=9)

# ------------------------------------------
# SUBPLOT 2: LOG-LOG SCALE
# ------------------------------------------
ax2.loglog(
    dx_log,
    T_ideal_log,
    "k--",
    linewidth=1.5
)
ax2.loglog(
    dx_log,
    T_trunc_log,
    color="crimson",
    linewidth=2.2
)

# Horizontal line for truncation limit
ax2.axhline(
    y=T_total,
    color="gray",
    linestyle="-.",
    linewidth=1.2
)

# --- Slope Triangle Annotation ---
x1_tri, x2_tri = 0.05, 0.4
y1_tri = A * (x1_tri ** (-gamma))
y2_tri = A * (x2_tri ** (-gamma))

# Draw triangle legs
ax2.plot([x1_tri, x2_tri], [y1_tri, y1_tri], color="royalblue", lw=1.5)
ax2.plot([x2_tri, x2_tri], [y1_tri, y2_tri], color="royalblue", lw=1.5)

# Annotate symbolic slope
ax2.text(
    np.sqrt(x1_tri * x2_tri),
    y1_tri * 1.25,
    r"$\Delta \log |x - x_c|$",
    ha="center",
    fontsize=9,
    color="royalblue",
)
ax2.text(
    x2_tri * 1.15,
    np.sqrt(y1_tri * y2_tri),
    r"Slope $= -\gamma$",
    va="center",
    fontsize=10,
    fontweight="bold",
    color="royalblue",
)

ax2.set_xlabel(r"Distance to critical point $|x - x_c|$", fontsize=12)
ax2.set_ylabel(r"$\overline{T}(x)$", fontsize=12)
ax2.set_title("Log-Log Scale", fontsize=13, fontweight="bold")

# Replace numerical ticks with symbolic markers
ax2.set_xticks([])
ax2.set_yticks([T_total])
ax2.set_yticklabels([r"$T_{\text{total}}$"], fontsize=12)

ax2.grid(True, which="both", linestyle=":", alpha=0.5)
ax2.legend(loc="lower left", fontsize=9)

# Overall Title
plt.suptitle(
    r"Power-Law Scaling mean time in between burst $\overline{T}(x)$ for some parameter $x$",
    fontsize=14,
    fontweight="bold",
    y=0.98,
)

plt.show()