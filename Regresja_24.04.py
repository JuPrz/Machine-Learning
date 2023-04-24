import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder 
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
df_x = pd.read_csv("Salary Data.csv" , sep = "," , encoding = 'utf-8')
print(df_x)
df_y=df_x['Salary']
df_x=df_x.drop('Salary')
# najpierw zmienne kategoryczne na numeryczne
# potem dzielimy zbiór na testowy i treningowy 
# budujemy model regresji 
# policzyć regresję liniową 

# Zmiana zmiennych kategorycznych na numeryczne 
cols = ['Gender', 'Education Level', 'Job Title','Age','Salary']
# Encode labels of multiple columns at once
df_x[cols] = df_x[cols].apply(LabelEncoder().fit_transform)

print(df_x.head())




# Standaryzacja danych numerycznych 
scaler = MinMaxScaler()
df_cleaned = df_x[cols]
col_float = ['Gender', 'Education Level', 'Job Title']
for i in df_cleaned[col_float]:
    df_cleaned[i] =scaler.fit_transform(df_cleaned[[i]])
print(df_cleaned.head())



# # potem dzielimy zbiór na testowy i treningowy 
# X_train, X_test, y_train, y_test = train_test_split( df_x,df_y, test_size=0.33, random_state=42)