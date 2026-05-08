import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jn_zeros

n = np.arange(1, 20)


plt.scatter(n, 1/(jn_zeros(1, 19)/jn_zeros(1, 1)[0])**2)
plt.show()