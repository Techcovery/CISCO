#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:26:06 2020

@author: nikmural
"""

import quandl as Quandl, math

from sklearn import preprocessing, model_selection as cross_validation
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from matplotlib import style
import datetime


style.use('ggplot')

#Use Pandas to populate the data frame
df = Quandl.get("WIKI/AMZN",api_key='tJbUksb_TeZXtwAgDeia')
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
#print(df.head(5))


#Calculate percentages
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0


#Predict closing values
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
forecast_col = 'Adj. Close'

#print(df.head(5))


#Replace any NAN so that data is not lost
#df.fillna(value=-99999, inplace=True)

#Replace any NAN with mode
for column in df.columns:
    df[column].fillna(df[column].mode()[0], inplace=True)
    
forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

# X is the input data and excludes labels to be predicted
X = np.array(df.drop(['label'], 1))
#Preprocessing gets values between -1 and 1
X = preprocessing.scale(X)

#Input for which prediction is needed
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)

#Output to be predicted
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

#Linear Regression
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print('Confidence Linear Regression:', confidence)

#Use Lasso regression classifier
clf = Lasso(alpha=0.1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print('Confidence Lasso:', confidence)

#Use Ridge regression classifier
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
clf = Ridge(alpha=1.0)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print('Confidence Ridge:', confidence)


#Use Polynomial regression classifier
poly = PolynomialFeatures(degree=3)
X_train_ = poly.fit_transform(X_train)
clf = LinearRegression()
clf.fit(X_train_, y_train)

X_test_ = poly.fit_transform(X_test)
confidence = clf.score(X_test_, y_test)
print('Confidence polynomial fit for degree 3:', confidence)

#Use Principal Component analysis
pca = PCA(0.95)
X_train_ = pca.fit_transform(X_train)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train_, y_train)

X_test_ = pca.fit_transform(X_test)
confidence = clf.score(X_test_, y_test)
print('Confidence PCA(0.95):', confidence)

print('Polynomial fit of degree 3 gives the highest confidence.')

#Commenting out plotting data
'''
#Predict for the forecast set
forecast_set = clf.predict(X_lately)
df['Forecast'] = np.nan
#Predict for every day after the last day in training set
#print(df.head(0))

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
#Plot the graph for visualizayion
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

#Store the model
import pickle
with open('linearregression.pickle','wb') as f:
    pickle.dump(clf, f)
'''
