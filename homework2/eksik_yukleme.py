# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 18:34:14 2021

@author: hulus
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("eksikveriler.csv")

#print(veriler)

boy = veriler[["boy"]]
#print(boy)

boy_kilo = veriler[["boy","kilo"]]
#print(boy_kilo)

x=10

class insan:
    boy= 90
    def kosmak(self,b):
        return b+10

ali = insan()
#print(ali.boy)
#print(ali.kosmak(90))


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

Yas = veriler.iloc[:,1:4].values
print(Yas)

imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)














