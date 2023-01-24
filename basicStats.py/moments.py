import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp

vals = np.random.normal(0,0.5, 10000)

plt.hist(vals, 50)
plt.show()


print("First Moment (Mean) : ", np.mean(vals))
print("Second Moment (Varience) : ", np.var(vals))
print("Third Moment (skew) : ",sp.skew(vals))
print('Forth Moment (Kurtosis) : ', sp.kurtosis(vals))
