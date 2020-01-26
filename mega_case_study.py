# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset 

dataset=pd.read_csv("Credit_Card_Applications.csv")
X=dataset.iloc[:,:-1].values
# to check the approved and un approved this un suprvised no label
y=dataset.iloc[:,-1].values 

# Feature Scalling 

from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
X=sc.fit_transform(X)

# Training the SOM

# we will use another developer impelmentation for the moment (minisom.py file)
from minisom import MiniSom

# the dimntions of the map (10*10) 
# the input length (the number of features)
som=MiniSom(x=10,y=10,input_len=15,sigma=1.0,learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X,num_iteration=100)

# visualizaing the results

from pylab import bone,pcolor,colorbar,plot,show
bone()
pcolor(som.distance_map().T) # for the pcolor take the transpose
colorbar()
markers=['o','s'] # o means circle and S means square
colors=['r','g'] # green and red colors
for i,x in enumerate(X):  # i is the idx for customer and x is the vector of [i] (the row)
    w=som.winner(x) # get the winning node of the customer with vector x
    plot(w[0]+0.5
         ,w[1]+0.5
         ,markers[y[i]]
         ,markeredgecolor=colors[y[i]]
         ,markerfacecolor='None'
         ,markersize=10
         ,markeredgewidth=2)
show()

# finding the frauds 

mappings=som.win_map(X)
# concatenate the white nodes (farthest distnace on the map)
frauds=np.concatenate((mappings[(8,7)],mappings[(4,1)]),axis=0)

 