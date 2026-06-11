from equations import mu_spectrum
import matplotlib.pyplot as plt
import numpy as np

N = 20

plt.plot(range(N), np.diff(np.sqrt(mu_spectrum(N+1)))/2/np.pi)
plt.show()