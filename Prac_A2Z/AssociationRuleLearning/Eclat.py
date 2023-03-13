import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = '/home/piyush/Projects/MLPract/dataSets/data_A2Z/'
FileName = 'Market_Basket_Optimisation.csv'

dataSet = pd.read_csv(path + FileName, header = None, )
dataSet

transactions =[]
for i in range(0,len(dataSet)):
    transactions.append([str(dataSet.values[i,j]) for j in range(0,len(dataSet.columns))])

from apyori import apriori
rules = apriori(transactions= transactions,
        min_support = 0.003, min_confidence = 0.2,min_lift = 3,
        min_length=2, max_length=2)

results = list(rules)

def inspect(results):
    Product1 = [tuple(result[2][0][0])[0] for result in results]
    Product2 = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(Product1,Product2,supports))

resultsTable = inspect(results)

resultsDataFrame = pd.DataFrame(resultsTable, columns=['Product1 ', 'Product2','Support'])
resultsDataFrame

resultsDataFrame.nlargest(n = 10, columns='Support')
