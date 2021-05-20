# polynommial regression
    
#Tmproting the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset with pandas
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, [3,4]].values


#dendogram method - finding optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))

plt.title('dendrogram')
plt.xlabel('customers')
plt.ylabel('distance')
plt.show()

#fitting hierarchical clustering
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean' ,linkage = 'ward')
yhc = hc.fit_predict(x)

#visualising clustes
plt.scatter(x[yhc == 0,0], x[yhc == 0, 1], s = 100, c= 'red', label='cluster 1')
plt.scatter(x[yhc == 1,0], x[yhc == 1, 1], s = 100, c= 'blue', label='cluster 2')
plt.scatter(x[yhc == 2,0], x[yhc == 2, 1], s = 100, c= 'green', label='cluster 3')
plt.scatter(x[yhc == 3,0], x[yhc == 3, 1], s = 100, c= 'yellow', label='cluster 4')
plt.scatter(x[yhc == 4,0], x[yhc == 4, 1], s = 100, c= 'violet', label='cluster 5')
plt.title('clusters')
plt.xlabel('income')
plt.ylabel('spending score')
plt.legend()
plt.show()
