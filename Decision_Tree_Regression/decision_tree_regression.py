# Importing the libraries
from sklearn.tree import DecisionTreeRegressor
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
    r'D:/VS Code/Workspace/Machine-Learning--Self-taught-/Decision_Tree_Regression/Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Decision Tree Regression model on the whole dataset
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x, y)

# Predict a new result
pred = regressor.predict([[6.5]])
# print(pred)

# Visualising
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color='red')
plt.plot(x_grid, regressor.predict(
    x_grid), color='blue')
plt.title('(Decision Tree Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()
