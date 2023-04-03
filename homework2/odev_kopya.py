# -*- coding: utf-8 -*-
"""
Created on Sat May  1 17:20:15 2021

@author: hulus
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing

veriler = pd.read_csv("odev_tenis.csv")

#Encoder

veriler2 = veriler.apply(preprocessing.LabelEncoder().fit_transform)

hava = veriler2.iloc[:,:1].values
ohe = preprocessing.OneHotEncoder()
hava = ohe.fit_transform(hava).toarray()
print(hava)

#DataFrame dönüşümleri

sonuc_hava = pd.DataFrame(data = hava , index= range(14), columns = ["overcast", "rainy", "sunny"])

s1 = pd.concat([sonuc_hava,veriler.iloc[:,1:3]], axis=1)
s2 = pd.concat([s1,veriler2.iloc[:,-2:-1]], axis=1)
print(s2)
oyun = veriler2.iloc[:,-1:]
print(oyun)
#VERİLERİN EĞİTİM VE TEST İÇİN BÖLÜNMESİ

from sklearn.model_selection import train_test_split

t_train, t_test, z_train, z_test = train_test_split(s2,oyun,test_size=0.33,random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(t_train,z_train)

z_pred = regressor.predict(t_test)

#Backward elemination

import statsmodels.api as sm

J = np.append(arr = np.ones((14,1)).astype(int), values=s2, axis=1)

J_l =s2.iloc[:,[0,1,2,3,4,5]].values
J_l =np.array(J_l, dtype=float)
model = sm.OLS(oyun,J_l).fit()
print(model.summary())
"""
J_l =s2.iloc[:,[0,1,2,4,5]].values
J_l =np.array(J_l, dtype=float)
model = sm.OLS(sonuc_oyun,J_l).fit()
print(model.summary())

J_l =s2.iloc[:,[0,1,2,5]].values
J_l =np.array(J_l, dtype=float)
model = sm.OLS(sonuc_oyun,J_l).fit()
print(model.summary())
"""




