# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 18:34:14 2021

@author: hulus
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("veriler.csv")

print(veriler)

boy = veriler[["boy"]]
print(boy)

boy_kilo = veriler[["boy","kilo"]]
print(boy_kilo)

x=10

