# polynommial regression
    
#Tmproting the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values


#splitting data
"""from sklearn.cross_validation import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state=0)
"""
#scaling
from sklearn.preprocessing import StandardScaler
scx = StandardScaler()
scy = StandardScaler()
x = scx.fit_transform(x)
y = scy.fit_transform(y)

#fitting regression
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(x,y)



ypred = scy.inverse_transform(regressor.predict(scx.transform(np.array([[6.5]]))))

#visualising
xgrid = np.arange(min(x),max(x), 0.1)
xgrid = xgrid.reshape((len(xgrid), 1)) #ładniejsze krągłosci

plt.scatter(x, y, color = 'red')
plt.plot(xgrid, regressor.predict((xgrid)), color = 'purple')
plt.title('Salary and Position (SVR)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show