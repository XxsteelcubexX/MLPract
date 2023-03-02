import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = '/home/piyush/Projects/MLPract/dataSets/data_A2Z/'
fileName = 'Position_Salaries.csv'

dataSet = pd.read_csv(path+fileName)
X = dataSet.iloc[:,1:-1].values
Y = dataSet.iloc[:,-1].values
print(X,Y, sep='\n#############\n')

# decision tree model below on whole data set

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)

regressor.fit(X,Y)
regressor.predict([[6.5]])

def plotDTRGraph_HD():
    X_grid = np.arange(min(X), max(X),0.1)
    X_grid = X_grid.reshape((len(X_grid),1))
    plt.scatter(X,Y , color = 'Red')
    plt.plot(X_grid, regressor.predict(X_grid).reshape(-1,1), color = 'Blue')
    plt.title("Decision Tree Regression Model")
    plt.xlabel("Position Level")
    plt.ylabel('Salary')

plotDTRGraph_HD()
