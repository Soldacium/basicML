# thompson sampling
    
#Tmproting the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#thompson sampling(look for formula online)
import random
n = 10000 #number of data points
d = 10 #number of ads
adsel =[] #selected ads
numofreward1 = [0] *d   #how many times the algorithm got 1(+ setting the table to 10 zeros([0]*d))
numofreward0 = [0] *d #how many times the algorithm got 0
totalreward = 0

for N in range(0,n):
    maxuppbound = 0 #max upper bound
    ad = 0
    maxrand = 0
    for i in range(0, d):
        randombeta = random.betavariate(numofreward1[i] + 1,numofreward0[i] + 1)
        if randombeta > maxrand:
            maxrand = randombeta
            ad = i
    adsel.append(ad)
    reward = dataset.values[N, ad]
    if reward == 1:
        numofreward1[ad] = numofreward1[ad] +1
        
    else:
        numofreward0[ad] = numofreward0[ad] +1
    totalreward = totalreward + reward
    
#visualising
plt.hist(adsel)
plt.title('Histogram')
plt.xlabel('ads')
plt.ylabel('num of ads selected')
    
     
       