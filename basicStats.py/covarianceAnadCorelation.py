import numpy as np
import matplotlib.pyplot as plt

def de_mean(x):
    xmean = np.mean(x)
    return [xi - xmean for xi in x]

def covariance(x,y):
    n = len(x)
    return np.dot(de_mean(x), de_mean(y)) / (n-1)

pageSpeeds = np.random.normal(3.0,1.0,1000)
purchaseAmount = np.random.normal(50.0,10.0, 1000)

plt.scatter(pageSpeeds,purchaseAmount)

covariance(pageSpeeds, purchaseAmount)

#############################################

purchaseAmount = np.random.normal(50.0,10.0, 1000) / pageSpeeds

plt.scatter(pageSpeeds, purchaseAmount)

covariance(pageSpeeds,purchaseAmount)

def correlation(x,y):
    stddevx = x.std()
    stddevy = y.std()
    return covariance(x,y)/stddevx/stddevy

correlation(pageSpeeds, purchaseAmount)

# easy way

np.corrcoef(pageSpeeds, purchaseAmount)
# a perfect correlation example below

purchaseAmount = 100 - pageSpeeds * 4
plt.scatter(pageSpeeds, purchaseAmount)
correlation(pageSpeeds, purchaseAmount)
