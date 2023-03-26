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

N = 50
d = 10
adsSelected = []
numbers_of_selections = [0]*d
sumsOfRewards=[0]*d
totalReward=0

for n in range (0,N):

    ad = 0
    maxUpperbond = 0
    for i in range(0,d):
        print(f'row {n}, column {i}')
        if sumsOfRewards[i] > 0:
            averageReward = sumsOfRewards[i] / numbers_of_selections[i]
            deltaI = math.sqrt(3/2 *math.log(n+1)/ numbers_of_selections[i])
            UpperBound = averageReward + deltaI
            print('Inner for loop',averageReward,deltaI,UpperBound,sep = '  ||  ',end = '\n********\n' )

        else:
            UpperBound = 1e400
        if UpperBound > maxUpperbond :
            maxUpperbond = UpperBound
            ad = i


    adsSelected.append(ad)
    numbers_of_selections[ad] += 1
    reward = dataSet.values[n,ad]
    if sumsOfRewards[ad] == 0:
        sumsOfRewards[ad] += 1


    sumsOfRewards[ad] = sumsOfRewards[ad] + reward
    totalReward = totalReward + reward

    print('adsSelected || ',adsSelected,end = '\n++++++++\n')
    print(numbers_of_selections,reward,sumsOfRewards,totalReward,sep=' | ',end='\n\n%%%%%%%%%%%\n\n')
    print(f'{n} Row Ends here')
    #input("Going into next row in dataSet \n Press Enter \n\n|#########|")

def plotHist():
    plt.hist(adsSelected)
    plt.title('Histogram of ads Selection')
    plt.xlabel('Ads')
    plt.ylabel('numbers_of_selections')
    plt.show()

plotHist()
