# Importing the libraries
from sklearn.ensemble import RandomForestRegressor
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
    r'D:/VS Code/Workspace/Machine-Learning--Self-taught-/Random_Forrest_Regression/Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the random forrest model on whole dataset
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(x, y)

# Predict a new result
pred = regressor.predict([[6.5]])

# Visualising
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color='red')
plt.plot(x_grid, regressor.predict(
    x_grid), color='blue')
plt.title('(Random Forrest Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()
