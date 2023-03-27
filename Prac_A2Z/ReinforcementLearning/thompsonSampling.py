# in this dataset each row is considered as round and each row has a cost to it in real world.
# each AD has fix conversion rate (Assumption with Reinforced learning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

path = '/home/piyush/Projects/MLPract/dataSets/data_A2Z/'
fileName = 'Ads_CTR_Optimisation.csv'

dataSet = pd.read_csv(path + fileName)
dataSet.head()

# implementing Thompson Sampling

import random
N = 10000
d = 10
adsSelected = []
numbers_of_rewards_1 = [0]*d
numbers_of_rewards_0 = [0]*d
totalReward = 0


for n in range(0,N):
    ad = 0
    max_random = 0
    for  i in range(0,d):
        randomBeta = random.betavariate(numbers_of_rewards_1[i]+1,
                            numbers_of_rewards_0[i]+1)

        if randomBeta > max_random:
            max_random = randomBeta
            ad = i

    adsSelected.append(ad)
    reward = dataSet.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad]+1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad]+1

    totalReward = totalReward + reward

def plotHist():
    plt.hist(adsSelected)
    plt.title('Histogram of ads Selection')
    plt.xlabel('Ads')
    plt.ylabel('numbers_of_selections')
    plt.show()

plotHist()
