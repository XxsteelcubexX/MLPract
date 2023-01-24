import numpy as np
import matplotlib.pyplot as plt

vals = np.random.normal(0,0.5,10000)

plt.hist(vals,50)
plt.show()

np.percentile(vals,50)

np.percentile(vals,90)

np.percentile(vals,20)
