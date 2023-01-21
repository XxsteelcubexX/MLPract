import numpy as np

# we are creating some fake income data, centered around 27,000 with a normal distribution
#and standard deviation of 15,000 with 10,000 data points.
#Then, compute the mean(avg) - it should be close 27,000.

incomes = np.random.normal(27000,15000,10000)
np.mean(incomes)

# we can segment the income data into 5o buckets, and plot it as a histogram.

%matplotlib inline
import matplotlib.pyplot as plt
plt.hist(incomes,50)
#plt.show()
np.median(incomes)
# lets add Elon Musk into the misc
incomes = np.append(incomes,[100000000000])
#the median will not change but mean will change
print(np.mean(incomes),np.median(incomes))

ages = np.random.randint(18,high = 90, size = 500)
from scipy import stats
stats.mode(ages)
