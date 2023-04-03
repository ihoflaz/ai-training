# -*- coding: utf-8 -*-
"""
Created on Sat May  8 00:10:47 2021

@author: hulus
"""

#1.KÜTÜPHANELER
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import statsmodels.api as sm
#2.1 VERİ YÜKLEME
veriler = pd.read_csv("maaslar_yeni.csv")

x = veriler.iloc[:,2:5]
y = veriler.iloc[:,5:]
X = x.values
Y = y.values

#LinearRegression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#Polynomial Regression
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

print(lin_reg2.predict(poly_reg.fit_transform([[12]])))
print(lin_reg2.predict(poly_reg.fit_transform([[5.5]])))

print(lin_reg.predict([[6]]))
print(lin_reg.predict([[5.5]]))

































