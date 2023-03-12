import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = '/home/piyush/Projects/MLPract/dataSets/data_A2Z/'
fileName = 'Mall_Customers.csv'

dataSet = pd.read_csv(path+fileName)
dataSet.head()
X = dataSet.iloc[:,[3,4]].values
X

# using the dendrogram to find the optimal number of clusters
def dendrogram_Plot():
    from scipy.cluster import hierarchy
    plt.figure(figsize=(14,7))
    dendrogram = hierarchy.dendrogram(hierarchy.linkage(X, method = 'ward'))
    plt.title('dendrogram')
    plt.xlabel('Customers')
    plt.ylabel('Eculidean Distances')

dendrogram_Plot()
# here we have two optimal cluster from this chart 3 and 5 clusters

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity='euclidean', linkage="ward")
Y_hc = hc.fit_predict(X)
def hc_graph():
    plt.figure(figsize=(14,7))
    plt.scatter(X[Y_hc == 0,0],X[Y_hc == 0,1],s =100, label = 'Cl1')
    plt.scatter(X[Y_hc == 1,0],X[Y_hc == 1,1],s =100, label = 'Cl2')
    plt.scatter(X[Y_hc == 2,0],X[Y_hc == 2,1],s =100, label = 'Cl3')
    plt.scatter(X[Y_hc == 3,0],X[Y_hc == 3,1],s =100, label = 'Cl4')
    plt.scatter(X[Y_hc == 4,0],X[Y_hc == 4,1],s =100, label = 'Cl5')
    plt.xlabel('Annual Income k$')
    plt.ylabel('Spending Score (1-100)')
    plt.title('Hierarchical Clustering')


hc_graph()
