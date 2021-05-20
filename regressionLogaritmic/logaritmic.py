# log regression
    
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
scy= StandardScaler()
ytrain = scy.fit_transform(ytrain)

#fitting
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain,ytrain)

#Predicting
ypred = classifier.predict(xtest)

#confusion matrix/patrzymy jak dobrze przewiduje
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, ypred)


#visualising
from matplotlib.colors import ListedColormap
xser, yset = xtrain, ytrain
x1, x2 = np.meshgrid(np.arange(start = xset[:,0].min() - 1, stop = xset[:,0].max() +1,step=0.01),
         np.meshgrid(np.arange(start = xset[:,1].min() - 1, stop = xset[:,1].max() +1,step=0.01))
     plt.contourf(x1, x2, classifier(np.array(x1.ravel() x2.ravel()]).T)
                  alpha = 0.75, cmap = ListedColormap(('red', 'green')))
     plt.xlim(x1.min(), x1.max())
     plt.ylim(x2.min(), x2.max())
     for i, j in enumerate(np.unique(yset)):
         plt.scatter(xset[yest == j, 0], xset[yset == j,1]),
         c = ListedColormap(('red', 'green'))(i),label=j)
                                        
plt.title('Salary and Position (RFR)')
plt.xlabel('age')
plt.ylabel('Salary')
plt.legend()
plt.show