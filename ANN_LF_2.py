# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 20:59:14 2024

@author: brand
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Data
dataset = pd.read_csv('C2DB_X.csv')

X = dataset.iloc[:, 2:1558].values
y = dataset.iloc[:, 1558].values

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Scaling the variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


#ANN model
classifier = Sequential()

#Input layer and first hiden layer
classifier.add(Dense(units = 778, kernel_initializer = 'uniform', activation = 'relu', input_dim = 1556))

#2nd hiden layer
classifier.add(Dense(units = 778, kernel_initializer = 'uniform', activation = 'relu'))

#3rd hiden layer
classifier.add(Dense(units = 778, kernel_initializer = 'uniform', activation = 'relu'))

#4th hiden layer
#classifier.add(Dense(units = 778, kernel_initializer = 'uniform', activation = 'relu'))

#5th hiden layer
#classifier.add(Dense(units = 778, kernel_initializer = 'uniform', activation = 'relu'))

#6th hiden layer
#classifier.add(Dense(units = 778, kernel_initializer = 'uniform', activation = 'relu'))

#7th hiden layer
#classifier.add(Dense(units = 778, kernel_initializer = 'uniform', activation = 'relu'))

# Output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation='relu'))


#ANN compilation
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mae'])

#Adjust the ANN to the train test
classifier.fit(X_train, y_train, batch_size = 10, epochs = 50)
y_pred = classifier.predict(X_test)  


loss, mae = classifier.evaluate(X_test, y_test)

print("Mean Squared Error (MSE):", loss)
print("Mean Absolute Error (MAE):", mae)


#Create Plot
import matplotlib.pyplot as plt

# PBE vs test results
y_min = min(min(y_test), min(y_pred))
y_max = max(max(y_test), max(y_pred))

plt.scatter(y_test, y_pred, color='royalblue', label='Predicted bandgaps')
plt.plot([y_min, y_max], [y_min, y_max], color='black', linestyle='--')  

plt.title('Artificial NN (LD data)')
plt.xlabel('PBE (eV)')
plt.ylabel('Prediction (eV)')
plt.legend()  
plt.grid(True)

plt.xlim(y_min, y_max)
plt.ylim(y_min, y_max)

plt.show()





