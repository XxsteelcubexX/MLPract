import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble.tree import RandomForestRegressor

path = '/home/piyush/Projects/MLPract/dataSets/data_A2Z/'
fileName = 'EnergyData.csv'

dataSet = pd.read_csv(path+fileName)
print(dataSet.head())
dataSet.isnull().any()
X = dataSet.iloc[:,:-1].values
Y = dataSet.iloc[:,-1].values
print(X,Y, sep='\n****************************\n')

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

# Pending add all the models and then selecting the best One

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
