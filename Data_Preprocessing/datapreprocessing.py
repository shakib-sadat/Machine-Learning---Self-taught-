# Importing the libraries
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
# read_csv= import dataset
dataset = pd.read_csv(
    r'D:/VS Code/Workspace/Machine-Learning--Self-taught-/Data_Preprocessing/Data.csv')
# x = features iloc=collect the index : = taking range, first take row, then take range -1 to take all the columns except the target column
x = dataset.iloc[:, :-1].values
# y =  dependent variable vector, -1 = last column
y = dataset.iloc[:, -1].values
# print(x)
# print(y)


# Data cleaning(missing data)
# replacing 'nan' =empty values replacing missing values with mean = strategy='mean'
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# fit method will look for numeric missing values in defined range in this case index 1 and 2
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])  # replace the new columns
# print(x)

# Encode categorical data
# creating the categorical column into numerical separate columns. for example 3 different countries will be 3 different columns
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')  # remainder='passthrough' is used to not include the exsisting numerical columns.
# will return new matrix, making numpy array becasue training set will expect it.
x = np.array(ct.fit_transform(x))
# print(x)

lblen = LabelEncoder()
y = lblen.fit_transform(y)  # convert into binary.
# print(y)

# Spliting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, train_size=0.8, random_state=1)

# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)

# Feature Scaling -> to avoid dominance (Standardisation (more used, transforms the values to +3 to -3, don't have to apply in dummy variable/ columns that've been created by ctegoriacl values. ), Normalisation)
standard_scaler = StandardScaler()
x_train[:, 3:] = standard_scaler.fit_transform(x_train[:, 3:])
# we have to apply the same scaler for prediction. so we don't need to use fit here becasue it'll create a different scaler.
x_test[:, 3:] = standard_scaler.transform(x_test[:, 3:])
print(x_train)
print(x_test)
