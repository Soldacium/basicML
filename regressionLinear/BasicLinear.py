#linear regression
  
#Tmproting the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#splitting data
from sklearn.cross_validation import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 1/3, random_state=0)

#scaling
"""from sklearn.preprocessing import StandardScaler
scx = StandardScaler()
xtrain = scX.fit_transform(xtrain)
xtest = scX.transform(xtest)
scy = StandardScaler()
ytrain = scy.fit_transform(ytrain)"""

#fitting
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

#prediction
ypred = regressor.predict(xtest)

#visualising training
plt.scatter(xtrain, ytrain, color = 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title('Salary and Experiance (Training)')
plt.xlabel('Y/o Experience')
plt.ylabel('Salary')
plt.show

#visualising test
plt.scatter(xtest, ytest, color = 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title('Salary and Experiance (Test)')
plt.xlabel('Y/o Experience')
plt.ylabel('Salary')
plt.show
