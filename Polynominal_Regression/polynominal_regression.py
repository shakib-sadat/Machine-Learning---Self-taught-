# Importing the libraries
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from dataclasses import replace
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Dataset
dataset = pd.read_csv(
    r'D:/VS Code/Workspace/Machine-Learning--Self-taught-/Polynominal_Regression/Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training Linear Regression on the whole model
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Training Polynominal on the whole model
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)  # Transformed features with powers
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# Visualising the linear regression
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg.predict(x), color='blue')
plt.title("Linear Regression")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

# Visualising the polynominal regression
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg_2.predict(x_poly), color='blue')
plt.title("Polynomial Regression")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color='red')
plt.plot(x_grid, lin_reg_2.predict(
    poly_reg.fit_transform(x_grid)), color='blue')
plt.title('(Polynomial Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
print(lin_reg.predict([[6.5]]))  # 2d array of the predicting result

# Predicting a new result with Polynomial Regression
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))
