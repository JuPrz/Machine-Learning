import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder 
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,classification_report 



#zaczytanie danych z pliku
df_data = pd.read_csv('C:\\Users\\Julia\\Desktop\\Multimedia\\Technologie multimedialne\\Machine earning\\Machine-Learning\\cleaned_data.csv', sep = ',')
print(df_data)

exclude_filter = ~df_data.columns.isin(['Unnamed: 0','Credit_Score'])
pca = PCA().fit(df_data.loc[:, exclude_filter])
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.gcf().set_size_inches(7, 5)
#plt.show()


pca = PCA(svd_solver='full', n_components=0.95)
principal_components = pca.fit_transform(df_data.loc[:, exclude_filter])
principal_df = pd.DataFrame(data=principal_components)
principal_df.head()
print(principal_df.head())
# Liczba cech 14

X_train, X_test, y_train, y_test = train_test_split(principal_df, df_data['Credit_Score'], test_size=0.33, random_state=42)
print(X_train.shape)
#kod

# model regresji 

clf = LogisticRegression(random_state=100)
clf.fit(X_train, y_train) #<- supervised learning
print(clf.predict(X_test))
# predict_proba(X)
y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
clas = y_test.unique()

ConfusionMatrixDisplay(cm).plot()
plt.show()

rep = classification_report(y_test, y_pred)
print(rep)
print(cm)

Accuracy = ((2488+10037+476)/(4186+106+504+1815+70+3366+2488+10037+476))


Precision_1 = (10037)/(4186+10037+3366)
Recall_1 = (10037)/(18150+10037+504)

print(Precision_1)
print(Recall_1)