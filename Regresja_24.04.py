from sklearn import linear_model
from tabulate import tabulate
from sklearn.preprocessing import PolynomialFeatures
from sklearn import tree
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder 
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
df = pd.read_csv("Salary Data.csv" , sep = "," , encoding = 'utf-8')
print(df)

# # Zmiana zmiennych kategorycznych na numeryczne 
#  Standaryzacja danych numerycznych 

columns = ['Gender', 'Education Level', 'Job Title']
df[columns] = df[columns].apply(LabelEncoder().fit_transform)

df = df.dropna(inplace=False)

X = df.drop('Salary', axis=1).values
y = df['Salary'].values

# potem dzielimy zbiór na testowy i treningowy 
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

regr = linear_model.LinearRegression()

regr.fit(x_train, y_train)
predictions = regr.predict(x_test)
print("Regression score:", regr.score(x_test, y_test))

# Lasso regression with alpha=0.1
lasso = linear_model.Lasso(alpha=0.1)
lasso.fit(x_train, y_train)
predictions_lasso = lasso.predict(x_test)

print("Lasso\nR-squared:", lasso.score(x_test, y_test))
print("MAE:", np.mean(np.abs(predictions_lasso - y_test)))

# Polynomial regression 3rd degree
poly = PolynomialFeatures(degree=3, include_bias=False)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.fit_transform(x_test)

regr_poly = linear_model.LinearRegression()
regr_poly.fit(x_train_poly, y_train)
predictions_poly = regr_poly.predict(x_test_poly)

print("Poly\nR-squared:", regr_poly.score(x_test_poly, y_test))
print("MAE:", np.mean(np.abs(predictions_poly - y_test)))

# Decision tree
regr_tree = tree.DecisionTreeRegressor(max_depth=5)
regr_tree.fit(x_train, y_train)
predictions_tree = regr_tree.predict(x_test)

print("Regression Tree\nR-squared:", regr_tree.score(x_test, y_test))
print("MAE:", np.mean(np.abs(predictions_tree - y_test)))

table = [['','Lasso','Polynomial','Tree'],
         ['R-squared',lasso.score(x_test, y_test),regr_poly.score(x_test_poly, y_test),regr_tree.score(x_test, y_test)],
         ['MAE',np.mean(np.abs(predictions_lasso - y_test)),np.mean(np.abs(predictions_poly - y_test)),np.mean(np.abs(predictions_tree - y_test))]]

scores = tabulate(table, headers='firstrow')
with open('score_table.txt', 'w') as f:
    f.write(scores)

