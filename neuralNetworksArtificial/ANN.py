# regrossion random tree
    
#Tmproting the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoderx1 = LabelEncoder()
x[:, 1] = labelencoderx1.fit_transform(x[:,1])
labelencoderx2 = LabelEncoder()
x[:, 2] = labelencoderx2.fit_transform(x[:,2])
onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()
x = x[:,1:]


#splitting data
from sklearn.cross_validation import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state=0)

#scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)

#importing keras and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#initialising the ANN(sequence of layers)
classifier = Sequential()

#adding hidden layer
classifier.add(Dense(output_dim = 6,init= 'uniform',activation = 'relu',input_dim=11))

#adding second hidden layer
classifier.add(Dense(output_dim = 6,init= 'uniform',activation = 'relu'))

#adding output layer
classifier.add(Dense(output_dim = 1,init= 'uniform',activation = 'sigmoid'))

#compiling
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#fitting ANN
classifier.fit(xtrain,ytrain, batch_size =10, nb_epoch = 100)

#predicting
ypred = classifier.predict(xtest)
ypred2 = (ypred>0.5)

#Confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, ypred2)