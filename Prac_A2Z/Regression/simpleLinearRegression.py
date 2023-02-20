import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
path = '/home/piyush/Projects/MLPract/dataSets/data_A2Z/'
fileName = 'Salary_Data.csv'

dataset = pd.read_csv(path+fileName)
dataset
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
print(X,Y)
#dataset.plot()

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, random_state=0)
print(X_train,X_test, Y_tarin, Y_test, sep='\n++++\n')

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train,Y_train)
predictedValues = regressor.predict(X_test)
Y_test

def plotGraphTrain():
    plt.scatter(X_train, Y_train, color = 'red')
    plt.plot(X_train,regressor.predict(X_train), color = 'Blue')
    plt.title('Salary VS Expirence (Training set)')
    plt.xlabel('Years of Exprience')
    plt.ylabel('Salary')
    plt.show()

def plotGraphTest():
    plt.scatter(X_test, Y_test, color = 'red')
    plt.plot(X_train,regressor.predict(X_train), color = 'Blue')
    plt.title('Salary VS Expirence (Test set)')
    plt.xlabel('Years of Exprience')
    plt.ylabel('Salary')
    plt.show()

plotGraphTrain()
plotGraphTest()


print(regressor.coef_)
print(regressor.intercept_)
