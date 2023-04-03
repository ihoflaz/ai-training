# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 18:34:14 2021

@author: hulus
"""
#1.KÜTÜPHANELER

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#2.VERİ ÖNİŞLEME

#2.1 VERİ YÜKLEME
veriler = pd.read_csv("satislar.csv")

#TEST
print(veriler)


aylar = veriler[["Aylar"]]
print(aylar)

satislar = veriler[["Satislar"]]
print(satislar)

"""
x=10

class insan:
    boy= 90
   def kosmak(self,b):
        return b+10

ali = insan()
print(ali.boy)
print(ali.kosmak(90))

#EKSİK VERİLER
#sci - kit learn

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

Yas = veriler.iloc[:,1:4].values
print(Yas)

imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)

#ENCODER   KATEGORİK -> NUMERIC

ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
print(ulke)


ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

#NUMPY DİZİLERİNİN DATAFRAME DÖNÜŞÜMÜ

sonuc = pd.DataFrame(data = ulke , index= range(22), columns = ["fr", "tr", "us"])
print(sonuc)

sonuc2 = pd.DataFrame(data = Yas , index = range(22), columns = ["boy", "kilo", "yas"])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = cinsiyet , index= range(22), columns=["cinsiyet"])
print(sonuc3)

#DATAFRAME BİRLEŞTİRME

s = pd.concat([sonuc,sonuc2], axis=1)
print(s)

s2 = pd.concat([s,sonuc3], axis=1)
print(s2)
"""


#VERİLERİN EĞİTİM VE TEST İÇİN BÖLÜNMESİ

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(aylar,satislar,test_size=0.33,random_state=0)

#VERİLERİN ÖLÇEKLENMESİ
"""
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
"""
#MODEL İNŞASI (LINEAR REGRESSION)

from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(x_train,y_train)

tahmin = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train,y_train)

plt.plot(x_test,lr.predict(x_test))

plt.title("Aylara Göre Satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")



