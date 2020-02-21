# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 23:20:53 2020

@author: fjesudha
"""

import pandas as pd
import numpy as np

from sklearn import preprocessing, model_selection as cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

ad = pd.read_csv(r"C:\Users\fjesudha\Desktop\Advertising.csv", index_col=0)
clt = LinearRegression()

x = ad['Newspaper'].values.reshape(-1,1)
sales = ad['Sales'].values.reshape(-1,1)
clt.fit(x, sales)
predicted_sales =clt.predict(x)


print("The linear model is: Y = {:.5} + {:.5}X".format(clt.intercept_[0], clt.coef_[0][0]))
#print(features, projection)

#yX = np.column_stack((ad['TV'], ad['Radio'], ad['Newspaper']))
plt.scatter(x, sales, c='black')
plt.plot(x, predicted_sales, c='blue')
plt.show()

#X = np.column_stack((ad['TV'], ad['Radio'], ad['Newspaper']))
X = np.column_stack((ad['TV'], ad['Radio']))
y = ad['Sales']
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())
