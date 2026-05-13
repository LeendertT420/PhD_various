import numpy as np
import warnings
warnings.filterwarnings('ignore')  # Suppress all warnings

verbose = True
np.set_printoptions(precision=4)


def rotate(v):
    # assumes (x1, y1, x2, y2, z)
    # returns (x-, y-, x+, y+, z)
    return [(v[0]-v[2])/2, (v[1]-v[3])/2, (v[0]+v[2])/2, (v[1]+v[3])/2, v[4]]


# -----------------------------
# bifurcation boundaries
# -----------------------------
def lower_boundary(d):
    s = np.sqrt(d**2 - 3)
    return -1/27 * (s - 2*d)**2 * (s + d)


def upper_boundary(d):
    s = np.sqrt(d**2 - 3)
    return  1/27 * (s + 2*d)**2 * (s - d)

# -----------------------------
# lasing threshold
# -----------------------------
def extract_real_entries(arr, epsilon=1e-12):
    """
    Return all entries whose imaginary part is smaller than epsilon.
    Returned values are converted to real floats.
    """
    arr = np.asarray(arr, dtype=complex)

    mask = np.abs(arr.imag) < epsilon

    return arr.real[mask]


def lasing_threshold(d, t, m, g1, g2, epsilon=1e-6):
    if verbose: print('CALCULATING LASING THRESHOLD')

    b1 = -(m + 1)**2*(g1*m + g2)/t**3
    b2 = (g1**4*m**2*t**2 + g1**3*g2*m**2*t**2 + g1**3*g2*m*t**2 + g1**3*m**2*t - g1**3*m*t + 2*g1**2*g2**2*m*t**2 + g1**2*g2*m**2*t + 2*g1**2*g2*m*t - g1**2*g2*t + g1**2*m**3*t**2 - 4*g1**2*m**2*t**2 - g1**2*m*t**2 - g1**2*m + g1*g2**3*m*t**2 + g1*g2**3*t**2 - g1*g2**2*m**2*t + 2*g1*g2**2*m*t + g1*g2**2*t - g1*g2*m**3*t**2 - 5*g1*g2*m**2*t**2 - g1*g2*m**2 - 5*g1*g2*m*t**2 - g1*g2*t**2 - g1*g2 + 2*g1*m**3*t - 2*g1*m**2*t + g2**4*t**2 - g2**3*m*t + g2**3*t - g2**2*m**2*t**2 - 4*g2**2*m*t**2 - g2**2*m + g2**2*t**2 - 2*g2*m*t + 2*g2*t - m**3*t**2 + 2*m**2*t**2 - m*t**2)/t**4
    b3 = (g1**4*g2**2*m*t**2 + g1**4*g2*m**2*t**3 + g1**4*g2*m*t - g1**4*m**2*t**2 + g1**3*g2**3*m*t**2 + g1**3*g2**3*t**2 + g1**3*g2**2*m**2*t**3 + 2*g1**3*g2**2*m*t + g1**3*g2**2*t - 3*g1**3*g2*m*t**2 + g1**3*g2*m - g1**3*m**2*t**3 - g1**3*m**2*t + g1**2*g2**4*t**2 + g1**2*g2**3*m*t + g1**2*g2**3*t**3 + 2*g1**2*g2**3*t + 2*g1**2*g2**2*m**2*t**2 - 8*g1**2*g2**2*m*t**2 + g1**2*g2**2*m + 2*g1**2*g2**2*t**2 + g1**2*g2**2 + g1**2*g2*m**3*t**3 - 5*g1**2*g2*m**2*t**3 - 3*g1**2*g2*m*t**3 - 9*g1**2*g2*m*t + 2*g1**2*g2*t - g1**2*m**3*t**2 + 2*g1**2*m**2*t**2 - g1**2*m*t**2 + g1*g2**4*t**3 + g1*g2**4*t - 3*g1*g2**3*m*t**2 + g1*g2**3 - 3*g1*g2**2*m**2*t**3 + 2*g1*g2**2*m**2*t - 5*g1*g2**2*m*t**3 - 9*g1*g2**2*m*t + g1*g2**2*t**3 + 2*g1*g2*m**3*t**2 - 2*g1*g2*m**2*t**2 + 2*g1*g2*m**2 - 2*g1*g2*m*t**2 - 4*g1*g2*m + 2*g1*g2*t**2 + 2*g1*g2 - g1*m**3*t**3 - g1*m**3*t + 2*g1*m**2*t**3 + 2*g1*m**2*t - g1*m*t**3 - g1*m*t - g2**4*t**2 - g2**3*m*t**3 - g2**3*t - g2**2*m**2*t**2 + 2*g2**2*m*t**2 - g2**2*t**2 - g2*m**3*t**3 + 2*g2*m**2*t**3 - g2*m**2*t - g2*m*t**3 + 2*g2*m*t - g2*t)/t**4
    b4 = -g1*g2*(g1*t + t**2 + 1)*(g2*t + m*t**2 + 1)*(g1**2*m + g1*g2*m + g1*g2 + g2**2 + m**2 - 2*m + 1)/t**4

    dL_sols = np.roots([b1, b2, b3, b4])
    dL_sols = np.real(dL_sols[np.isreal(dL_sols)]) # get all real solutions

    if verbose: print('dL solutions', len(dL_sols), dL_sols)
    
    N = 2
    z_sols = []

    for sol in dL_sols:
        E = d**2 - N*sol*(N*sol + 2)
        z_sols.append(( -d*(N*sol + 1) + np.sqrt(E) ) / (N*(N*sol + 2)) )
        z_sols.append(( -d*(N*sol + 1) - np.sqrt(E) ) / (N*(N*sol + 2)) )

    z_sols = np.array(z_sols)
    if verbose: print(f'\tz solutions shape:{np.shape(z_sols)}')

    thresholds = z_sols * ((N*z_sols + d)**2 + 1)
    print(f'thresholds:{np.shape(thresholds)}')
    print(f'thresholds:{np.shape(filter_arrays(thresholds))}')

    alphas_sorted = sorted(filter_arrays(thresholds), key=lambda a: np.min(a))
    print(alphas_sorted)
    return alphas_sorted


def filter_arrays(arr_list):
    """
    Remove arrays that:
    - are entirely negative
    - consist only of NaN values
    """
    filtered = []

    for arr in arr_list:
        arr = np.asarray(arr)

        # skip arrays with only NaNs
        if np.all(np.isnan(arr)):
            continue

        # skip arrays that are entirely negative
        if np.all(arr < 0):
            continue

        filtered.append(arr)

    return filtered



def limit_cycle_bifurcation_lower(g, t):
    return (-2*g**3*t**2 - 3*g**2*t - g + t - np.sqrt(g*(4*g**5*t**4 + 12*g**4*t**3 - 8*g**3*t**4 + 13*g**3*t**2 - 20*g**2*t**3 + 6*g**2*t - 4*g*t**4 - 16*g*t**2 + g - 4*t**3 - 4*t)))/(t*(2*g*t + 1))



def limit_cycle_bifurcation_upper(g, t):
    return (-2*g**3*t**2 - 3*g**2*t - g + t + np.sqrt(g*(4*g**5*t**4 + 12*g**4*t**3 - 8*g**3*t**4 + 13*g**3*t**2 - 20*g**2*t**3 + 6*g**2*t - 4*g*t**4 - 16*g*t**2 + g - 4*t**3 - 4*t)))/(t*(2*g*t + 1))



# -----------------------------
# fixed points
# -----------------------------
def z_star(a, d):
    roots = np.roots([4, 4*d, d**2 + 1, -a])
    roots = np.real(roots[np.isreal(roots)])
    print(f'roots: {roots}')
    return roots

def dL_star(z, d):
    return -2*z*(2*z + d) / ((2*z + d)**2 + 1)

# -----------------------------
# Jacobian
# -----------------------------
def Jacobian(z, d, t, m, g1, g2):
    dL = dL_star(z, d)
    J = np.array([
            [0,      1,       0,      0,       0     ],
            [-1,   -g1, 0,      0,       1     ],
            [0,      0,       0,      1,       0     ], 
            [0,      0,       -m, -g2, m     ],
            [dL/t, 0,       dL/t, 0,       -1/t]
        ])
    if verbose: print(f'Jacobian:{J}')
    return J


def compute_eigs(a, d, t, m, g1, g2):
    roots = z_star(a, d)
    eigvals = []
    eigvecs = []
    if verbose: print('EIGENVALUES AND EIGENVECTORS:')
    for i, root in enumerate(roots):
        vals, vecs = np.linalg.eig(Jacobian(root, d, t, m, g1, g2))
        eigvals.append(vals)
        eigvecs.append(vecs)
        print(f'\troot {i}')
        if verbose:
            for j, (val, vec) in enumerate(zip(vals, vecs)):
                print(f'\t\tvalue {j}:{val}')
                print(f'\t\tvector {j}:{vec}')

    return roots, eigvals, eigvecs



# -----------------------------
# SYSTEM
# -----------------------------
def system(t, y, a, d, tau, m, g1, g2):
    x1, v1, x2, v2, z = y
    dx1dt = v1
    dv1dt = -g1 * v1 - x1 + z
    dx2dt = v2
    dv2dt = -g2 * v2 - m*(x2 - z)
    dzdt = a / ((x1+x2+d)**2+1)/tau - z/tau
    return [dx1dt, dv1dt, dx2dt, dv2dt, dzdt]



def project_onto_plane(x, v1, v2):
    """
    Project vector x onto the plane spanned by v1 and v2.

    Parameters:
        x, v1, v2 : array-like (shape: (n,))
    
    Returns:
        projection of x onto span{v1, v2}
    """
    # Stack vectors as columns of A (n x 2 matrix)
    A = np.column_stack((v1, v2))
    
    # Compute projection: A (A^T A)^{-1} A^T x
    ATA_inv = np.linalg.inv(A.T @ A)
    projection = A @ ATA_inv @ A.T @ x
    
    return projection