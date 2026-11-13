import numpy as np
from scipy.signal import hilbert
from scipy.signal import savgol_filter, hilbert

from tqdm import tqdm

from equations import *

def remove_slow_offset_auto(xs, dt, freqs, poly=3, period_factor=8):
    """
    Automatically choose Savitzky–Golay window per mode
    based on estimated oscillation frequency.
    """
    N_modes, N_t = xs.shape
    xs_detrended = np.zeros_like(xs)
    trends = np.zeros_like(xs)
    windows = []

    for i in range(N_modes):
        f = freqs[i]

        if f == 0:
            window = 101
        else:
            period_samples = int(1 / (f * dt))
            window = period_factor * period_samples

        window = max(window, poly + 2)
        if window % 2 == 0:
            window += 1
        window = min(window, N_t - (1 - N_t % 2))

        trend = savgol_filter(xs[i], window_length=window, polyorder=poly)

        xs_detrended[i] = xs[i] - trend
        trends[i] = trend
        windows.append(window)

    return xs_detrended, trends, np.array(windows), freqs


def get_phase(t, x):
    analytic = hilbert(x)
    return np.unwrap(np.angle(analytic))


def get_mode_locking_order_parameter(xs_array):
    """
    Computes an AMPLITUDE-WEIGHTED Mode-Locking Order Parameter R_ML(t) in [0, 1].
    This ignores phase noise from weak/zero-amplitude modes.
    xs_array: shape (N_modes, N_time_steps)
    """
    # Get complex analytic signals
    zs = hilbert(xs_array, axis=1)
    
    # Calculate z_{n+1} * conjugate(z_n)
    # The angle of this product is exactly (phi_{n+1} - phi_n)
    # The magnitude is A_{n+1} * A_n (this provides the natural amplitude weighting)
    adjacent_cross_terms = zs[1:] * np.conj(zs[:-1])
    
    # Coherent sum (magnitudes of the summed vectors)
    coherent_sum = np.abs(np.sum(adjacent_cross_terms, axis=0))
    
    # Incoherent sum (sum of the magnitudes)
    incoherent_sum = np.sum(np.abs(adjacent_cross_terms), axis=0)
    
    # Safe division
    R_ML_weighted = np.divide(
        coherent_sum, 
        incoherent_sum, 
        out=np.zeros_like(coherent_sum), 
        where=incoherent_sum!=0
    )
    
    return R_ML_weighted



def calculate_mode_locked_fraction(
    ranges_list_threshold, classification, T_sim, T_eval
):
  """Calculates the total time fraction spent in mode-locked intervals."""
  if classification != 'CHAOTIC':
    print('modelocking but not chaotic!!!!')
    return 0.0

  if (
      ranges_list_threshold is None
      or len(ranges_list_threshold) == 0
      or T_eval <= 0
  ):
    return 0.0

  total_ml_time = 0.0

  for t_start, t_stop in ranges_list_threshold:
    if t_start is not None and t_stop is not None:
      total_ml_time += t_stop - t_start
    elif t_start is not None and t_stop is None:
      total_ml_time += T_eval + T_sim - t_start

  return min(1.0, total_ml_time / T_eval)



def get_mode_locking_order_parameter2(xs_array):
    """
    Computes an AMPLITUDE-WEIGHTED Mode-Locking Order Parameter R_ML(t) in [0, 1].
    This ignores phase noise from weak/zero-amplitude modes.
    xs_array: shape (N_modes, N_time_steps)
    """
    # Get complex analytic signals
    zs = hilbert(xs_array, axis=1)

    coherent_sum = np.zeros(np.shape(xs_array)[1]).astype('complex128')
    incoherent_sum = np.zeros(np.shape(xs_array)[1]).astype('complex128')

    for i in range(np.shape(xs_array)[0]):
        for j in range(np.shape(xs_array)[0]):
            if i != j:
                crossterm = zs[i] * np.conj(zs[j])

                coherent_sum += crossterm

                incoherent_sum += np.abs(crossterm)
    
    # Calculate z_{n+1} * conjugate(z_n)
    # The angle of this product is exactly (phi_{n+1} - phi_n)
    # The magnitude is A_{n+1} * A_n (this provides the natural amplitude weighting)

    
    # Safe division
    R_ML_weighted = np.divide(
        np.abs(coherent_sum), 
        incoherent_sum, 
        out=np.zeros_like(coherent_sum), 
        where=incoherent_sum!=0
    )
    
    return R_ML_weighted
