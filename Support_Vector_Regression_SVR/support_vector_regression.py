# Importing the libraries
from sklearn.svm import SVR
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
    r'D:/VS Code/Workspace/Machine-Learning--Self-taught-/Support_Vector_Regression_SVR/Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling -> to avoid dominance (Standardisation (more used, transforms the values to +3 to -3, don't have to apply in dummy variable/ columns that've been created by ctegoriacl values. ), Normalisation)

y = y.reshape(len(y), 1)

standard_scaler_x = StandardScaler()
standard_scaler_y = StandardScaler()

x = standard_scaler_x.fit_transform(x)
y = standard_scaler_y.fit_transform(y)

# print(x)
# print(y)

# Training the SVR model on the whole dataset
regressor = SVR(kernel='rbf')
regressor.fit(x, y)

# Predicting a new result
standard_scaler_y.inverse_transform(
    regressor.predict(standard_scaler_x.transform([[6.5]])))

# Visualising SVR results
plt.scatter(standard_scaler_x.inverse_transform(
    x), standard_scaler_y.inverse_transform(y), color='red')
plt.plot(standard_scaler_x.inverse_transform(
    x), standard_scaler_y.inverse_transform(regressor.predict(x)), color='blue')
plt.title("SVR Regression")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()
