import sklearn
import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
# sklearn.linear_model.Ridge
# sklearn.linear_model.Lasso

df = pd.read_csv("Salary-Data.csv" , sep = "," , encoding = 'utf-8')


# Zmiana zmiennych kategorycznych na numeryczne 
cols = ['Gender', 'Education Level', 'Job Title']

# Encode labels of multiple columns at once
df[cols]=df[cols].apply(LabelEncoder().fit_transform)

print(f"data:\n", 
      df.head())

# Standaryzacja danych numerycznych 
scaler = MinMaxScaler()
col_stand = ['Age','Gender','Education Level', 'Job Title', 'Years of Experience']
for i in df[col_stand]:
    df[i] = scaler.fit_transform(df[[i]])
print(f"cleanded data:\n", 
      df.head())
X = df[col_stand]
y = df['Salary']

print(f"y:\n", 
      y.head())
print(f"X:\n", 
      X.head())

# podział zbioru na testowy i treningowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# regresja liniowa 
linReg = LinearRegression().fit(X_train, y_train)
        # to zwraca R^2
linReg_score_train = linReg.score(X_train, y_train)
linReg_score_test = linReg.score(X_test, y_test)
  
y_predict = linReg.predict(X_test)
mean_squared_error(y_true=y_test, y_pred= y_predict)
 
print(f"linear:\n", 
      linReg_score_train)
# print(f"predict:\n", 
#       y_predict)

# Regresja liniowa LASSO
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)

lasso_score_train = lasso.score(X_train, y_train)
lasso_score_test = lasso.score(X_test, y_test)

print(f"lasso:\n", 
      lasso_score_train)

#Regresja wielomianowa stopnia 3
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

lin_reg = LinearRegression()
lin_reg.fit(X_train_poly, y_train)

poly_score_train = lin_reg.score(X_train_poly, y_train)
poly_score_test = lin_reg.score(X_test_poly, y_test)

print(f"poly:\n", 
      poly_score_train)

#Regresja z wykorzystaniem k-NN lub drzewa decyzyjnego
knn = KNeighborsRegressor()
knn.fit(X_train, y_train)

knn_score_train = knn.score(X_train, y_train)
knn_score_test = knn.score(X_test, y_test)

print(f"knn:\n", 
      knn_score_train)

tree = DecisionTreeRegressor(max_depth=5)
tree.fit(X_train, y_train)

tree_score_train = tree.score(X_train, y_train)
tree_score_test = tree.score(X_test, y_test)

print(f"tree:\n", 
      tree_score_train)




train_scores = [linReg_score_train,lasso_score_train, poly_score_train, knn_score_train, tree_score_train]
test_scores = [linReg_score_test, lasso_score_test, poly_score_test, knn_score_test, tree_score_test]
models = ['Linear Regression','Lasso', 'Polynomial Regression', 'k-NN', 'Decision Tree']

#Wykres z wunikami
plt.figure(figsize=(10, 5))
plt.plot(models, train_scores, label='Train Score')
plt.plot(models, test_scores, label='Test Score')
plt.xlabel('Model')
plt.ylabel('Score')
plt.title('Model Comparison')
plt.legend()
plt.show()

#Tabela wyników
results = pd.DataFrame({'Model': models,
                        'Train Score': train_scores,
                        'Test Score': test_scores})