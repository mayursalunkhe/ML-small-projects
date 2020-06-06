import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values  # : = all columns, :-1 = except last column
Y = dataset.iloc[:, 4].values


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy variable trap
X = X[:, 1:]

# Splitting dataset into traing and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

# Feature Scalling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting Multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the test set results
Y_pred = regressor.predict(X_test)

np.set_printoptions(precision=2)
print(np.concatenate((Y_pred.reshape(len(Y_pred), 1), Y_test.reshape(len(Y_test), 1)), 1))

## Building the optimal model using Backward Elimination
#import statsmodels.formula.api as sm
## add coloumn of ones as eqution is y = b0X0 + ... + bnXn X0 = 1
##X = np.append(arr = X, values = np.ones((50, 1)).astype(int), axis = 1)
## we need to add one column of ones for 50 rows,
## we want to add ones column in begining of X so invert it and looks like values = X
#X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
#
## X_opt will contain optimal metrix ie it contains dependent variable which have high impact of independent variable
#
#X_opt = X[:, [0, 1, 2, 3, 4, 5]]
#regressor_OLS = sm.ols()