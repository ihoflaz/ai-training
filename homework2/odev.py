# -*- coding: utf-8 -*-
"""
Created on Sat May  1 17:20:15 2021

@author: hulus
"""

import numpy as np
import pandas as pd

veriler = pd.read_csv("odev_tenis.csv")

derece_ve_nem = veriler.iloc[:,1:3]
"""print(derece_ve_nem)"""

#Hava
hava = veriler.iloc[:,0:1].values
print(hava)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

hava[:,0] = le.fit_transform(hava[:,0])
print(hava)


ohe = preprocessing.OneHotEncoder()
hava = ohe.fit_transform(hava).toarray()
print(hava)

#Rüzgar
ruzgar = veriler.iloc[:,-2:-1].values
print(ruzgar)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ruzgar[:,0] = le.fit_transform(ruzgar[:,0])

ohe = preprocessing.OneHotEncoder()
ruzgar = ohe.fit_transform(ruzgar).toarray()
print(ruzgar)

#Oyun
oyun = veriler.iloc[:,-1:].values
print(oyun)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

oyun[:,0] = le.fit_transform(oyun[:,0])

ohe = preprocessing.OneHotEncoder()
oyun = ohe.fit_transform(oyun).toarray()
print(oyun)

#DataFrame dönüşümleri

sonuc_hava = pd.DataFrame(data = hava , index= range(14), columns = ["overcast", "rainy", "sunny"])
print(sonuc_hava)

sonuc_ruzgar = pd.DataFrame(data = ruzgar[:,1] , index= range(14), columns = ["rüzgar"])
print(sonuc_ruzgar)

sonuc_oyun = pd.DataFrame(data = oyun[:,0] , index= range(14), columns = ["oyun"])
print(sonuc_oyun)

s1 = pd.concat([sonuc_hava,derece_ve_nem], axis=1)
print(s1)

s2 = pd.concat([s1,sonuc_ruzgar], axis=1)
print(s2)

#VERİLERİN EĞİTİM VE TEST İÇİN BÖLÜNMESİ

from sklearn.model_selection import train_test_split

t_train, t_test, z_train, z_test = train_test_split(s2,sonuc_oyun,test_size=0.33,random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(t_train,z_train)

z_pred = regressor.predict(t_test)

#Backward elemination

import statsmodels.api as sm

J = np.append(arr = np.ones((14,1)).astype(int), values=s2, axis=1)

J_l =s2.iloc[:,[0,1,2,3,4,5]].values
J_l =np.array(J_l, dtype=float)
model = sm.OLS(sonuc_oyun,J_l).fit()
print(model.summary())

J_l =s2.iloc[:,[0,1,2,4,5]].values
J_l =np.array(J_l, dtype=float)
model = sm.OLS(sonuc_oyun,J_l).fit()
print(model.summary())

J_l =s2.iloc[:,[0,1,2,5]].values
J_l =np.array(J_l, dtype=float)
model = sm.OLS(sonuc_oyun,J_l).fit()
print(model.summary())





