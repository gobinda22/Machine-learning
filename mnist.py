# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 20:53:46 2021

@author: Gobinda
"""

from sklearn.datasets import fetch_openml  #--For spliting the data 
from sklearn.linear_model import LinearRegression #---For train our model using LinearRegression
from sklearn.metrics import mean_squared_error


X,y = fetch_openml('mnist_784',version=1,return_X_y=(True), as_frame=(False))

x_train = X[:-30]
x_test = X[-20:]
y_train= y[:-30]
y_test = y[-20:]

print(x_train.shape,y_train.shape)

model = LinearRegression() 

model.fit(x_train,y_train)

pred = model.predict(x_test)

print(pred)

print("MSE:", mean_squared_error(y_test, pred))

print("Accuracy:", model.score(x_test, y_test))