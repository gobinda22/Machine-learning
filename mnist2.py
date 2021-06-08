# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 01:02:01 2021

@author: Gobinda
"""

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

mnist = fetch_openml(data_id=554)

print(type(mnist))
print(type(mnist.data), type(mnist.categories), type(mnist.feature_names), type(mnist.target))
print(mnist.data.shape, mnist.target.shape)
"""
#Preview some images
"""
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(mnist.data[0:5], 
                                           mnist.target[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
    plt.title('Training: ' + label, fontsize = 20);
"""    
#Split into traing dataset
#targets str to int convert
"""
X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.9 ,random_state=0)

print(X_train.shape, X_test.shape)

#Learning
model = LinearRegression() 

model.fit(X_train, y_train)

pred = model.predict(X_test)

print(pred)

print("MSE:", mean_squared_error(y_test, pred))

print("Accuracy:", model.score(X_test, y_test))

