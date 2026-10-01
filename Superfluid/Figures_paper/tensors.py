import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import j0, j1, jn_zeros
from scipy.integrate import quad

def compute_chi_tensor(N=10):
    """
    Computes the N x N x N tensor chi_ijk defined as:
    chi_ijk = 4 * integral_0^1 ( J0(zeta_i * r) * J0(zeta_j * r) * J0(zeta_k * r) 
                                / (J0(zeta_i) * J0(zeta_j) * J0(zeta_k)) ) * r dr
    where zeta_i are the zeros of J1(x) corresponding to Neumann boundary conditions.
    """
    # Zeros of J1(x) correspond to Neumann BCs J0'(zeta_i) = -J1(zeta_i) = 0
    zeros = jn_zeros(1, N)
    
    # Pre-evaluate J0(zeta_i) denominators
    j0_zeros = j0(zeros)
    
    chi = np.zeros((N, N, N))
    
    # Calculate symmetry-reduced tensor elements via numerical quadrature
    for i in range(N):
        for j in range(i, N):
            for k in range(j, N):
                zi, zj, zk = zeros[i], zeros[j], zeros[k]
                denom = j0_zeros[i] * j0_zeros[j] * j0_zeros[k]
                
                # Integrand definition
                integrand = lambda r: r * j0(zi * r) * j0(zj * r) * j0(zk * r)
                val, _ = quad(integrand, 0, 1, limit=100)
                
                chi_val = (4.0 / denom) * val
                
                # Assign all symmetric permutations
                chi[i, j, k] = chi_val
                chi[i, k, j] = chi_val
                chi[j, i, k] = chi_val
                chi[j, k, i] = chi_val
                chi[k, i, j] = chi_val
                chi[k, j, i] = chi_val
                
    return chi, zeros

# --- Compute Tensor ---
N_modes = 20
chi, zeros = compute_chi_tensor(N=N_modes)

# Set style
sns.set_theme(style="ticks", font_scale=1.1)
fig = plt.figure(figsize=(15, 10))

# --- Plot 1 & 2: Heatmap Slices for k=1 and k=2 ---
ax1 = fig.add_subplot(2, 2, 1)
sns.heatmap(chi[:, :, 0], annot=True, fmt=".2f", cmap="magma", ax=ax1,
            xticklabels=np.arange(1, N_modes+1), yticklabels=np.arange(1, N_modes+1))
ax1.set_title(r"$\chi_{ij1}$ Slice (Coupling with Mode $k=1$)", fontweight="bold")
ax1.set_xlabel("Mode Index $j$")
ax1.set_ylabel("Mode Index $i$")

ax2 = fig.add_subplot(2, 2, 2)
sns.heatmap(chi[:, :, 1], annot=True, fmt=".2f", cmap="magma", ax=ax2,
            xticklabels=np.arange(1, N_modes+1), yticklabels=np.arange(1, N_modes+1))
ax2.set_title(r"$\chi_{ij2}$ Slice (Coupling with Mode $k=2$)", fontweight="bold")
ax2.set_xlabel("Mode Index $j$")
ax2.set_ylabel("Mode Index $i$")

# --- Plot 3: Decay Profile of High Modes ---
ax3 = fig.add_subplot(2, 2, 3)
k_indices = np.arange(1, N_modes+1)
ax3.plot(k_indices, chi[0, 0, :], 'o-', label=r"$\chi_{1,1,k}$ (Pumped Mode Pair)", color="tab:blue", lw=2)
ax3.plot(k_indices, [chi[i, i, i] for i in range(N_modes)], 's--', label=r"$\chi_{k,k,k}$ (Self-Coupling)", color="tab:red", lw=2)
ax3.set_xlabel("Mode Index $k$")
ax3.set_ylabel(r"Overlap Value $\chi$")
ax3.set_title("Coupling Decay Across Modes", fontweight="bold")
ax3.set_xticks(k_indices)
ax3.grid(True, linestyle=":", alpha=0.6)
ax3.legend(frameon=True)

# --- Plot 4: Diagonal & Adjacent Cross-Coupling Terms ---
ax4 = fig.add_subplot(2, 2, 4)
adjacent_coupling = [chi[i, i, min(i+1, N_modes-1)] for i in range(N_modes)]
ax4.plot(k_indices, [chi[i, i, i] for i in range(N_modes)], 's-', color="darkred", label=r"Self-Interaction $\chi_{i,i,i}$")
ax4.plot(k_indices[:-1], adjacent_coupling[:-1], 'd--', color="teal", label=r"Nearest-Neighbor $\chi_{i,i,i+1}$")
ax4.set_xlabel("Mode Index $i$")
ax4.set_ylabel(r"Overlap Value $\chi$")
ax4.set_title("Self vs. Nearest-Neighbor Coupling Strengths", fontweight="bold")
ax4.set_xticks(k_indices)
ax4.grid(True, linestyle=":", alpha=0.6)
ax4.legend(frameon=True)

plt.tight_layout()
plt.show()