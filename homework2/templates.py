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

x = veriler.iloc[:,1:4].values
y = veriler.iloc[:,4:].values

#VERİLERİN EĞİTİM VE TEST İÇİN BÖLÜNMESİ

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=0)

#VERİLERİN ÖLÇEKLENMESİ

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(x_test)

print(y_pred)
print(np.ravel(y_test))





































