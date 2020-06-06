
# Simple linear Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values  # : = all columns, :-1 = except last column
Y = dataset.iloc[:, 1].values # index of dependent variable column


# Splitting dataset into traing and test set
# making train set - 20 and test set - 10 obs
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 1/3, random_state = 0)

# Feature Scalling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# no need to apply feature scalling

# Fitting Simple Linear Regression to Training set

from sklearn.linear_model import LinearRegression
# create object of LinearRegression class

regressor = LinearRegression()
regressor.fit(X_train,Y_train) # fit simple linear regressor to training set

# The machine here is Simple linear Regression
# we created machine regressor and made machine learn on our training set

# Predicting the test set results
Y_pred = regressor.predict(X_test)

# Visualising the traning set results

plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the test set results

plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()