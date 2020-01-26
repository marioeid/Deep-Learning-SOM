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
frauds=np.concatenate((mappings[(8,6)],mappings[(7,6)]),axis=0)
frauds=sc.inverse_transform(frauds)


# going from un suprvised deep learning to suprvised deep learning 

# creating matrix of features 

customers=dataset.iloc[:,1:].values 

# creating independent variable 

is_fraud=np.zeros(len(dataset))

for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds :
        is_fraud[i]=1


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)




                    
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# intializing the ANN
classifier=Sequential()

#Adding the input layer and the first hidden layer kernel intializer intialzies the weights intialy
# relu is the reqtifier function

# need to define the input in the first layer only
classifier.add(Dense(units=2,kernel_initializer='uniform',activation='relu',input_dim=15))


# adding the output layer
classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

# compiling the NN the algorithm u want to use is the optimizer (gradient descent)
# here we will use one of the stochastic gradient algorithms and it's name is Adam
# loss is the lost function in the gradient algo
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Fitting classifier to the Training set
# batch size size when to update weights 
# epochs number of iterations on gradient descenet
classifier.fit(customers,is_fraud,batch_size=1,epochs=2)
                          
                       
# Predicting the probabilites of frauds 
y_pred = classifier.predict(customers)
# iloc[0:1] cause concatenation will only work if both 2d arrays and that's how to make it 2d array
y_pred=np.concatenate((dataset.iloc[:,0:1].values,y_pred),axis=1)
y_pred=y_pred[y_pred[:,1].argsort()] # sort by the column of index 1 
