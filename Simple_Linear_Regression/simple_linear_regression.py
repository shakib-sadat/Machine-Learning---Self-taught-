# Importing the libraries
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
    r'D:/VS Code/Workspace/Machine-Learning--Self-taught-/Simple_Linear_Regression/Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Spliting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, train_size=0.8, random_state=0)

# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)

# Training the simple linear regression model on the training set
regressor = LinearRegression()
regressor.fit(x_train, y_train)  # call fit() to train the training set

# Predicting the test set result
y_pred = regressor.predict(x_test)  # y_pred contains predicted salaries.

# visualize the training set results
plt.scatter(x_train, y_train, color='purple')  # real data
plt.plot(x_train, regressor.predict(x_train), color='blue')  # predictive line
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# visualize the test set results
plt.scatter(x_test, y_test, color='red')  # real data
# predictive line, here we'll get the same line for both test and train set.
plt.plot(x_train, regressor.predict(x_train), color='green')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
