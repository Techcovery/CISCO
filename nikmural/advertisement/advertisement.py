# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection as cross_validation, svm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
import matplotlib.pyplot as plt
from matplotlib import style
import datetime

style.use('ggplot')

df = pd.read_csv("Advertising.csv", index_col=0)
#df.head(30)

df.plot()
plt.show()

#Replace any NAN with mode
for column in df.columns:
    df[column].fillna(df[column].mode()[0], inplace=True)
   
features = ['TV', 'Radio', 'Newspaper']
X = df[features]
#print(X)

y_label= 'Sales'
y = df[y_label]
#print(y)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)


clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print('Confidence Linear Regression:', confidence)

clf = Ridge(alpha=1.0)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print('Confidence Ridge:', confidence)

clf = Lasso(alpha=0.1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print('Confidence Lasso:', confidence)

poly = PolynomialFeatures(degree=2)
X_train_ = poly.fit_transform(X_train)
clf = LinearRegression()
clf.fit(X_train_, y_train)
#print(poly.get_feature_names())
X_test_ = poly.fit_transform(X_test)
confidence = clf.score(X_test_, y_test)
print('Confidence for Polynomial Regression with degree 2:', confidence)

poly = PolynomialFeatures(degree=3)
X_train_ = poly.fit_transform(X_train)
clf = LinearRegression()
clf.fit(X_train_, y_train)
#print(poly.get_feature_names())
X_test_ = poly.fit_transform(X_test)
confidence = clf.score(X_test_, y_test)
print('Confidence for Polynomial Regression with degree 3:', confidence)

print ('A polynomial fit of degree 3 gives the best result.')