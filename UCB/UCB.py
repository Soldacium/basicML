# Upper confidence boundry
    
#Tmproting the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#implementing upper confidence bound(look for formula online)
import math
n = 10000 #number of data points
d = 10 #number of ads
adsel =[] #selected ads
numofsel = [0] * d #number of selections
sumofrew = [0] * d #number of rewards
totalreward = 0

for N in range(0,n):
    maxuppbound = 0 #max upper bound
    ad = 0
    for i in range(0, d):
        if (numofsel[i]>0):
            avgrew = sumofrew[i]/numofsel[i] # average reward for ad nr. i
            deltai = math.sqrt(3/2 * math.log(N + 1)/numofsel[i]) #confidence interval
            uppbound = avgrew + deltai #upper bound
        else:
            uppbound = 1e400
        if uppbound > maxuppbound:
            maxuppbound = uppbound
            ad = i
    adsel.append(ad)
    numofsel[ad] = numofsel[ad]+1
    reward = dataset.values[N, ad]
    sumofrew[ad] = sumofrew[ad] + reward
    totalreward = totalreward + reward
    
#visualising
plt.hist(adsel)
plt.title('Histogram')
plt.xlabel('ads')
plt.ylabel('num of ads selected')
    
     
        