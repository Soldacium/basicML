# polynommial regression
    
#Tmproting the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values


#splitting data
from sklearn.cross_validation import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25, random_state=0)

#scaling
from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
xtrain = scX.fit_transform(xtrain)
xtest = scX.transform(xtest)

#fitting
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state=0)
classifier.fit(xtrain, ytrain)

#predictiong
ypred = classifier.predict(xtest)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(ytest,ypred)

#visualization
from matplotlib.colors import ListedColormap
xset, yset = xtrain, ytrain
x1, x2 = np.meshgrid(np.arange(start=xset[:,0].min() -1, stop= xset[:,0].max() +1,step=0.01),
                     np.arange(start=xset[:,1].min() -1, stop= xset[:,1].max() +1,step=0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
alpha=0.75, cmap = ListedColormap(('red','green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i,j in enumerate(np.unique(yset)):
    plt.scatter(xset[yset == j, 0], xset[yset == j,1],
                c = ListedColormap(('red','green'))(i),label = j)
    plt.title('K_NN (Training set)')
    plt.xlabel('age')
    plt.ylabel('salary')
    plt.legend()
    plt.show()