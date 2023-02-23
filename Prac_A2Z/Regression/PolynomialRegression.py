import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

path = '/home/piyush/Projects/MLPract/dataSets/data_A2Z/'
fileName = 'Position_Salaries.csv'

dataSet = pd.read_csv(path+fileName)
dataSet.head()

X = dataSet.iloc[:,1:-1].values
Y = dataSet.iloc[:,-1].values
print(X,Y)
dataSet.plot()

X_train,X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)



Lin_reg = LinearRegression()
Lin_reg.fit(X,Y)
Y_pred = Lin_reg.predict(X_test)

print(Y_pred, Y_test)

# Polynomial Regression start here
from sklearn.preprocessing import PolynomialFeatures
ploy_reg = PolynomialFeatures(degree = 2)# we will choose n in eqn here (or degree)
# above line creates the matrix feature
X_poly = ploy_reg.fit_transform(X)
X_poly

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,Y)
####################################
#########visualising
###################################

def plotLinearGraph():
    plt.scatter(X, Y, color = 'Red')
    plt.plot(X, Lin_reg.predict(X), color = 'blue')
    plt.title("LinearRegression Model")
    plt.xlabel("Position Level")
    plt.ylabel('Salary')

plotLinearGraph()

def plotPloyGraph():
    plt.scatter(X, Y, color = 'Red')
    plt.plot(X, lin_reg_2.predict(X_poly), color = 'gray')
    plt.plot(X, lin_reg_2.predict(ploy_reg.fit_transform(X)), color = 'Blue')
    plt.title("PolyNomial Regression Model")
    plt.xlabel("Position Level")
    plt.ylabel('Salary')

plotPloyGraph()

# retraing polynomial with higher power

ploy_reg = PolynomialFeatures(degree = 4)# we will choose n in eqn here (or degree)
# above line creates the matrix feature
X_poly = ploy_reg.fit_transform(X)
X_poly

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,Y)

plotPloyGraph()

# predicting the Salaries 

Lin_reg.predict([[6.5]])
polll = PolynomialFeatures(degree=4)
X_Polll = polll.fit_transform([[6.5]])
lin_reg_2.predict(X_Polll)
