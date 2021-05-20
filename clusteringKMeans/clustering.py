# k-means
    
#Tmproting the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, [3,4]].values


#elbow method - finding optimal numberclusters max_iter to max liczba wyszukañ, a n_init to max liczba powtorzen szukania
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('elbow method')
plt.xlabel('n clustrer')
plt.ylabel('wcss')
plt.show()

#applyinh kmeans
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
ykmeans = kmeans.fit_predict(x)

#visualising clustes
plt.scatter(x[ykmeans == 0,0], x[ykmeans == 0, 1], s = 100, c= 'red', label='cluster 1')
plt.scatter(x[ykmeans == 1,0], x[ykmeans == 1, 1], s = 100, c= 'blue', label='cluster 2')
plt.scatter(x[ykmeans == 2,0], x[ykmeans == 2, 1], s = 100, c= 'green', label='cluster 3')
plt.scatter(x[ykmeans == 3,0], x[ykmeans == 3, 1], s = 100, c= 'yellow', label='cluster 4')
plt.scatter(x[ykmeans == 4,0], x[ykmeans == 4, 1], s = 100, c= 'violet', label='cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s=300, c = 'cyan', label='centra')
plt.title('clusters')
plt.xlabel('income')
plt.ylabel('spending score')
plt.legend()
plt.show()
