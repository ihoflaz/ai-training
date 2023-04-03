# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 18:34:14 2021

@author: hulus
"""
#1.KÜTÜPHANELER

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

#2.VERİ ÖNİŞLEME

#2.1 VERİ YÜKLEME
veriler = pd.read_csv("maaslar.csv")

#DataFrame dilimleme (slice)
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

#Numpy dizi dönüşümü
X = x.values
Y = y.values

#LinearRegression
#Doğrusal model oluşturma
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#Polynomial Regression
#Doğrusal olmayan (nonlinear) model oluşturma

#2. Dereceden Polinom
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

#4. Dereceden Polinom
poly_reg2 = PolynomialFeatures(degree = 4)
x_poly = poly_reg2.fit_transform(X)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly,y)

#6. Dereceden Polinom
poly_reg3 = PolynomialFeatures(degree = 6)
x_poly = poly_reg3.fit_transform(X)
lin_reg4 = LinearRegression()
lin_reg4.fit(x_poly,y)
"""
#Görselleştirme
plt.scatter(X,Y, color="red")
plt.plot(X,lin_reg.predict(X), color="blue")
plt.show()

plt.scatter(X,Y,color="red")
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color="green")
plt.show()

plt.scatter(X,Y,color="red")
plt.plot(X,lin_reg3.predict(poly_reg2.fit_transform(X)),color="green")
plt.show()

plt.scatter(X,Y,color="red")
plt.plot(X,lin_reg4.predict(poly_reg3.fit_transform(X)),color="green")
plt.show()

#Predicts
print(lin_reg.predict([[12]]))
print(lin_reg.predict([[5.5]]))

print(lin_reg2.predict(poly_reg.fit_transform([[12]])))
print(lin_reg2.predict(poly_reg.fit_transform([[5.5]])))

print(lin_reg3.predict(poly_reg2.fit_transform([[1]])))
print(lin_reg3.predict(poly_reg2.fit_transform([[5.5]])))

print(lin_reg4.predict(poly_reg3.fit_transform([[1.5]])))
print(lin_reg4.predict(poly_reg3.fit_transform([[5.5]])))
"""


from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)

sc2 = StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(5,2)))

from sklearn.svm import SVR

svr_reg = SVR()
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli, color="m")
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color="c")
plt.show()

print(svr_reg.predict([[1.5]]))
print(svr_reg.predict([[5.5]]))

from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

Z = X + 0.5
K = X - 0.4
plt.scatter(X,Y,color="r")
plt.plot(X,r_dt.predict(X),color="b")

plt.plot(x,r_dt.predict(Z),color="g")
plt.plot(X,r_dt.predict(K),color="y")

plt.show()
print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))


from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators = 10,random_state=0)
rf_reg.fit(X,Y.ravel())

print(rf_reg.predict([[6.6]]))

plt.scatter(X, Y,color="r")
plt.plot(X,rf_reg.predict(X),color="b")

plt.plot(X,rf_reg.predict(Z),color="g")
plt.plot(X,rf_reg.predict(K),color="y")























