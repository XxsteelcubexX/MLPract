import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

path = '/home/piyush/Projects/MLPract/dataSets/data_A2Z/'
fileName = 'Social_Network_Ads.csv'
testFile = 'BreastCancer.csv'

#dataSet = pd.read_csv(path+fileName)
dataSet = pd.read_csv(path+testFile)
X = dataSet.iloc[:,:-1].values
Y = dataSet.iloc[:,-1].values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.25, random_state= 0 )

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10,criterion="entropy",random_state=0)
classifier.fit(X_train,Y_train)
#classifier.predict(sc.transform([[30,8700]]))

Y_pred = classifier.predict(X_test)

Con_Ys = np.concatenate((Y_pred.reshape(len(Y_pred),1),Y_test.reshape(len(Y_test),1)),1)
Con_Ys
cm = confusion_matrix(Y_test, Y_pred)
cm
accuracy_score(Y_test,Y_pred)
"""
def plottrainingSet():
    from matplotlib.colors import ListedColormap
    x_set, y_set = sc.inverse_transform(X_train), Y_train
    x1, x2 = np.meshgrid(
            np.arange(start=x_set[:,0].min()-10, stop=x_set[:,0].max()+10, step = 0.25),
            np.arange(start=x_set[:,1].min()-1000, stop =x_set[:,1].max()+1000, step = 0.25)
            )
    plt.contourf(x1,x2, classifier.predict(
            sc.transform(np.array([x1.ravel(), x2.ravel()]).T)).reshape(x1.shape),
            alpha = 0.75, cmap = ListedColormap(('red','green')))
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())
    for i,j in enumerate(np.unique(y_set)):
        plt.scatter(x_set[y_set == j, 0],x_set[y_set == j, 1],c=ListedColormap(('pink','green'))(i), label = j)
        plt.title('Random Forest (Train)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()


plottrainingSet()
# here predition is a straight line because it is a straight line model.

#for test ste

def plottestSet():
    from matplotlib.colors import ListedColormap
    x_set, y_set = sc.inverse_transform(X_test), Y_test
    x1, x2 = np.meshgrid(
            np.arange(start=x_set[:,0].min()-10, stop=x_set[:,0].max()+10, step = 0.25),
            np.arange(start=x_set[:,1].min()-1000, stop =x_set[:,1].max()+1000, step = 0.25)
            )
    plt.contourf(x1,x2, classifier.predict(
            sc.transform(np.array([x1.ravel(), x2.ravel()]).T)).reshape(x1.shape),
            alpha = 0.75, cmap = ListedColormap(('red','green')))
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())
    for i,j in enumerate(np.unique(y_set)):
        plt.scatter(x_set[y_set == j, 0],x_set[y_set == j, 1],c=ListedColormap(('pink','green'))(i), label = j)
        plt.title('Random Forest (Test)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()

plottestSet()
"""
