import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = '/home/piyush/Projects/MLPract/dataSets/data_A2Z/'
FileName = 'Market_Basket_Optimisation.csv'

dataSet = pd.read_csv(path + FileName, header = None, )
dataSet
# making format as list of transcations

len(dataSet)

dataSet.iloc[100]

len(dataSet.columns)
cout = 0
transcations = []
for i in range(0,len(dataSet)):
    #print(cout,end = '****************')
    #print(dataSet.iloc[i],)
    transcations.append([str(dataSet.values[i,j])for j in range(0,len(dataSet.columns))])
    cout += 1

print(transcations)


# training Apriori Model

# we are filtering above model based on support and confidence
# why 0.003 ?
# its because we want for product A->B, atleast 3 times a day. Given dataSaet is of week.
# therefore it becomes (3*7)/7501  = 0.002799627
# 0.2 -> is set based on the bussiness requirment
#min_lift = 3, generally lift below 3 are not that relevent.


# displaying the first results comming directly from the output

from apyori import apriori
rules = apriori(transactions= transcations,
        min_support = 0.003, min_confidence = 0.2,min_lift = 3,
        min_length=2, max_length=2)

results = list(rules)
results

def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidence = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lhs,rhs,supports,confidence,lifts))

resultsTable = inspect(results)

resultsDataFrame = pd.DataFrame(resultsTable, columns=[
'Left hand side', 'Right hand side','Support', 'Confidence','lifts'])
resultsDataFrame

resultsDataFrame.nlargest(n = 10, columns='lifts')
