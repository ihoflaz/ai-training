# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 18:34:14 2021

@author: hulus
"""
#1.KÜTÜPHANELER

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#2.VERİ ÖNİŞLEME

#2.1 VERİ YÜKLEME
veriler = pd.read_csv("veriler.csv")

x = veriler.iloc[5:,1:4].values
y = veriler.iloc[5:,4:].values

#VERİLERİN EĞİTİM VE TEST İÇİN BÖLÜNMESİ


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=0)

#VERİLERİN ÖLÇEKLENMESİ

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_train)

print(y_pred)
print(np.ravel(y_train))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_pred, labels=["e", "k"])
print(cm)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric="minkowski")
knn.fit(X_train,y_train)

y_pred = knn.predict(X_train)

cm = confusion_matrix(y_train,y_pred)
print(cm)


































