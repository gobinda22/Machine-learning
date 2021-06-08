# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 20:04:37 2021

@author: Gobinda
"""
from matplotlib import pyplot
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split #--For spliting the data 
from sklearn.linear_model import LinearRegression #---For train our model using LinearRegression
from sklearn.metrics import mean_squared_error


data = load_iris()
inputs = data['data'] 
targets = data['target']
#----spliting the data
 #--these are the variable used to split the data
x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, random_state=0,)
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

model.fit(x_train, y_train)

pred = model.predict(x_test)

print(pred)


print("weights :",model.coef_)

print("intercept :",model.intercept_)

print("MSE:", mean_squared_error( y_test, pred))

print("Accuracy:", model.score(x_test, y_test))
