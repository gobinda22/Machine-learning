# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 21:37:04 2021

@author: Gobinda
"""
from numpy.linalg import inv
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split #--For spliting the data 
from sklearn.linear_model import LinearRegression #---For train our model using LinearRegression

data = load_iris() #--load the data
inputs = data['data']      #---two part one is target and another one is data
targets = data['target']
#----spliting the data
x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, random_state=0,) #--these are the variable used to split the data

print("before spliting the data")
print(inputs.shape, targets.shape)

print("after spliting :")
print("train data :",x_train.shape, y_train.shape)
print("test data:",x_test.shape, y_test.shape)

"""
linear regression 
Y= XW

m- nmuber of example
d- input feature size
k- output feature size

X-m,d
Y-m,k
W-d,k

W= inv(X.T @ X) @ (X.T @ Y)
"""

model = LinearRegression()
model.fit(x_train , y_train )
def OwnPrediction(x_test ):
     W = inv(x_train.T @ x_train) @ (x_train.T @ y_train)
     Y = x_train @ (W)
     return Y
 
pred = OwnPrediction(x_test)

print(pred)

print("Accuracy:", model.score(x_test, y_test))

print("weights :",model.coef_)

print("intercept :",model.intercept_)