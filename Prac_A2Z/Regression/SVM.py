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
# because sclae of both variable is different
X = sc_X.fit_transform(X)
X
Y = sc_Y.fit_transform(Y)
Y

# Model creation

from sklearn.svm import SVR

regressor = SVR(kernel='rbf')
regressor.fit(X,Y)

# we nee to reverse the scaling used to input the model
# getting predictions
sc_Y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1,1))


def plotSVRGraph():
    plt.scatter(sc_X.inverse_transform(X), sc_Y.inverse_transform(Y), color = 'Red')
    plt.plot(sc_X.inverse_transform(X), sc_Y.inverse_transform(regressor.predict(X).reshape(-1,1)), color = 'Blue')
    plt.title("SVR Model")
    plt.xlabel("Position Level")
    plt.ylabel('Salary')

plotSVRGraph()

def plotSVRGraph_HD():
    X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)),0.1)
    X_grid = X_grid.reshape((len(X_grid),1))
    plt.scatter(sc_X.inverse_transform(X),sc_Y.inverse_transform(Y) , color = 'Red')
    plt.plot(X_grid, sc_Y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1,1)), color = 'Blue')
    plt.title("SVR Model")
    plt.xlabel("Position Level")
    plt.ylabel('Salary')

plotSVRGraph_HD()
