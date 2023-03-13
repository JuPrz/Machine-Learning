import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df_train = pd.read_csv("train.csv", sep ="," , encoding= 'UTF-8')  # NIE ROZDZIELA MI SEP PRZECINKÓW NA KOLUMNY
print(df_train)

# df_train.shape
# df_train.info()
# df_train.head()
# df_train.drop_duplicates(subset="ID", inplace=True)
# df_train.describe()

# for i in df_train.columns:
#     print (df_train[i].value_counts())
#     print ('*'*50)

# sns.countplot(df_train['Credit_Score']) 
# df_train.info()

# FeaturesToConvert = ['Age','Annual_Income','Num_of_Loan','Num_of_Delayed_Payment','Changed_Credit_Limit','Outstanding_Debt','Amount_invested_monthly','Monthly_Balance']


# # sprawdzanie czy nie ma błędów w danych 
# for feature in FeaturesToConvert:
#     uniques = df_train[feature].unique()
#     print('Feature:','\n', '\n', uniques,'\n','--'*40,'\n')


# # #  consumer ID też możemy usunąć
# # #  tam są różńe nazyw ale wszędzie daję df_train a nie jak ona ma df albo df test - MUSZE slec przeklejajac kod 

# # # usuń zbędne znaki '-’ , '_'
# for feature in FeaturesToConvert:
#     df_train[feature] = df_train[feature].str.strip('-_')

# # puste kolumny zastąp NAN
# for feature in FeaturesToConvert:
#     df_train[feature] = df_train[feature].replace({'':np.nan})

# # # zmiana typu danych ilościowych 
# for feature in FeaturesToConvert:
#     df_train[feature] = df_train[feature].astype('float64')

# # #uzupełnij braki średnią
# df_train['Monthly_Inhand_Salary']= df_train['Monthly_Inhand_Salary'].fillna(method='pad')

# # # zakoduj zmienne kategoryczne 
# from sklearn.preprocessing import LabelEncoder
# # # stwórz obiekt enkodera
# le = LabelEncoder()
# df_train.Occupation =le.fit_transform(df_train.Occupation)
# # sprawdź transformacje
# df_train.head()

# cols = ['Month','Credit_Mix','Payment_of_Min_Amount','Payment_Behaviour','Credit_Score','Type_of_Loan']
#  # Encode labels of multiple columns at once
# df_train[cols] = df_train[cols].apply(LabelEncoder().fit_transform)

# print(df_train.head().T) #transpozycja 