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
veriler = pd.read_csv("veriler.csv")

"""

#TEST
print(veriler)

boy = veriler[["boy"]]
print(boy)

boy_kilo = veriler[["boy","kilo"]]
print(boy_kilo)

x=10

class insan:
    boy= 90
   def kosmak(self,b):
        return b+10

ali = insan()
print(ali.boy)
print(ali.kosmak(90))

"""

"""

#EKSİK VERİLER
#sci - kit learn

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

"""

Yas = veriler.iloc[:,1:4].values
print(Yas)

"""
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)
"""
#ENCODER   KATEGORİK -> NUMERIC

#ÜLKE
ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
print(ulke)


ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

#CİNSİYET
c = veriler.iloc[:,-1:].values
print(c)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

c[:,-1] = le.fit_transform(veriler.iloc[:,-1])
print(c)


ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
print(c)

#NUMPY DİZİLERİNİN DATAFRAME DÖNÜŞÜMÜ

sonuc = pd.DataFrame(data = ulke , index= range(22), columns = ["fr", "tr", "us"])
print(sonuc)

sonuc2 = pd.DataFrame(data = Yas , index = range(22), columns = ["boy", "kilo", "yas"])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = c[:,:1] , index= range(22), columns=["cinsiyet"])
print(sonuc3)

#DATAFRAME BİRLEŞTİRME

s = pd.concat([sonuc,sonuc2], axis=1)
print(s)

s2 = pd.concat([s,sonuc3], axis=1)
print(s2)

#VERİLERİN EĞİTİM VE TEST İÇİN BÖLÜNMESİ

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s,sonuc3,test_size=0.33,random_state=0)

"""

#VERİLERİN ÖLÇEKLENMESİ

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

"""


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

boy = s2.iloc[:,3:4].values
print(boy)

sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]

veri = pd.concat([sol,sag], axis=1)

a_train, a_test, b_train, b_test = train_test_split(veri,boy,test_size=0.33,random_state=0)

r2 = LinearRegression()
r2.fit(a_train, b_train)

b_pred =r2.predict(a_test)

import statsmodels.api as sm

X = np.append(arr = np.ones((22,1)).astype(int), values=veri, axis=1)

X_l =veri.iloc[:,[0,1,2,3,4,5]].values
X_l =np.array(X_l, dtype=float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())

X_l =veri.iloc[:,[0,1,2,3,5]].values
X_l =np.array(X_l, dtype=float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())


























