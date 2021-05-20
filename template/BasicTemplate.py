# polynommial regression
    
#Tmproting the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


#splitting data
"""from sklearn.cross_validation import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state=0)
"""
#scaling
"""from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
Xtrain = scX.fit_transform(Xtrain)
xtest = scX.transform(xtest)
scy= StandardScaler()
ytrain = scy.fit_transform(ytrain)"""

#fitting
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x, y)

ypred = regressor.predict(x)

plt.scatter(x, y, color = 'red', edgecolors = 'green')
plt.plot(x, regressor.predict(x), color = 'purple')
plt.title('Salary and Position (Linear)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show

#fitting polynomial
from sklearn.preprocessing import PolynomialFeatures
regressor2 = PolynomialFeatures(degree = 4)
xpoly = regressor2.fit_transform(x)
regressor2.fit(xpoly, y)
regressorpoly = LinearRegression()
regressorpoly.fit(xpoly,y)

#visualising
xgrid = np.arange(min(x),max(x), 0.1)
xgrid = xgrid.reshape((len(xgrid), 1)) #ładniejsze krągło

plt.scatter(x, y, color = 'red', edgecolors = 'green')
plt.plot(x, regressorpoly.predict(regressor2.fit_transform(x)), color = 'purple')
plt.title('Salary and Position (Polynomial)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show