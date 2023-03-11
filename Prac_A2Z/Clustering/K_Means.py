import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
path = '/home/piyush/Projects/MLPract/dataSets/data_A2Z/'
fileName = 'Mall_Customers.csv'

dataSets = pd.read_csv(path+fileName)
print(dataSets.head())
dataSets.isnull().any()

X = dataSets.iloc[:,1:].values
print(X)
le = LabelEncoder()
X[:,0] = le.fit_transform(X[:,0])
print(X)
# Every thing here is correct ( good job Piyush !)
# we can do with more than 2 features too!.

# here we will select just 2 features only beacuse instructor wantss to show us the plot
# he can does that by exactly 2 features and hence we will take features only

X = X[:,[2,3]]
X

# using Elbow to find the optimal number of clusters
from sklearn.cluster import KMeans
WCSS = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init="k-means++",random_state = 42)
    kmeans.fit(X)

    WCSS.append(kmeans.inertia_)

print(WCSS)

def WCCS_graph():
    plt.plot(range(1,11),WCSS)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS Value')
    plt.show()

WCCS_graph()


kmeans = KMeans(n_clusters=5,init='k-means++', random_state=42)
# creating dependent variable
Y_kmeans = kmeans.fit_predict(X)

#visualizing
def kmeans_graph():
    plt.scatter(X[Y_kmeans == 0,0],X[Y_kmeans == 0,1],s =100, label = 'Cl1')
    plt.scatter(X[Y_kmeans == 1,0],X[Y_kmeans == 1,1],s =100, label = 'Cl2')
    plt.scatter(X[Y_kmeans == 2,0],X[Y_kmeans == 2,1],s =100, label = 'Cl3')
    plt.scatter(X[Y_kmeans == 3,0],X[Y_kmeans == 3,1],s =100, label = 'Cl4')
    plt.scatter(X[Y_kmeans == 4,0],X[Y_kmeans == 4,1],s =100, label = 'Cl5')
    plt.xlabel('Annual Income k$')
    plt.ylabel('Spending Score (1-100)')
    plt.title('K-Means')
    plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'Centroids')
kmeans_graph()
