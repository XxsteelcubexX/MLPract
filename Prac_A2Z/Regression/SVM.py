import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = '/home/piyush/Projects/MLPract/dataSets/data_A2Z/'
fileName = 'Position_Salaries.csv'

dataSet = pd.read_csv(path+fileName)
dataSet.head()
X = dataSet.iloc[:,1:-1].values
Y = dataSet.iloc[:,-1].values
print(X,Y,sep='\n&&&&\n')

"""here we don't have train and test split in the and hence
going to feature scaling step
We don't have a train, test split here because we want
to leaverage the maximum data possible in this example"""
#Feature Scaling
# We need to convert 1d data to 2d array data as model expects the data in 2d format
Y = Y.reshape(len(Y),1)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
# here we need to create two scaler object
X = sc_X.fit_transform(X)
X
Y = sc_Y.fit_transform(Y)
Y
