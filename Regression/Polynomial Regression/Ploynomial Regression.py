import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Polinomial regression model

# y = b0 + b1.X1 + b2.X1/\2 + b2.X1/\3 + ...
# create metrix of powered features

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# visulize linear reg
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Salary (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# visulize Poly reg
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Salary (Poly Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# visulize Poly reg High res
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid), color = 'blue')
plt.title('Salary (Randor forest tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


lin_reg.predict([[6.5]]) # input is array of 2D, first [ is row and second [ is column.

lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))