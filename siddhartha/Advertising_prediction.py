# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("Advertising.csv")
data.head()

# use multivarite linear regression to predict the 
# advetisement budgets on sales

# We will have only the "TV, Radio and NewsPaper" as the features

X = data.drop(['Sales', 'Unnamed: 0'], axis=1)
Y = data['Sales'].values.reshape(-1,1)

reg = LinearRegression()
reg.fit(X,Y)

print("The linear model is: Y = {:.5} + {:.5}*TV + {:.5}*radio + {:.5}*newspaper".format(reg.intercept_[0], 
      reg.coef_[0][0], reg.coef_[0][1], reg.coef_[0][2]))

# predict different spending on sales
predictionData = np.array([[230.1, 40.1, 70.0 ]])
sales_pred = reg.predict(predictionData)
print("Prediction for sales with the adverstisement spendings as [TV: 230.1, Radio: 40.1, News Paper: 70.0 ] :%s" %sales_pred)

confidence = reg.score(X, Y)
print('Confidence:', confidence)

# Train Test Split the model
from sklearn.model_selection import train_test_split
x_train , x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the test set results. We will try to predict the
# salary of the test set and will see how close the model predict the result.

y_pred = regressor.predict(predictionData)
print("Prediction with Train/Test/Split for [TV: 230.1, Radio: 40.1, News Paper: 70.0 ] :%s" %y_pred)
confidence = reg.score(x_train, y_train)
print('Confidence X_Train YTrain:', confidence)




## PCA approach
features = ['TV', 'Radio', 'Newspaper']

df = pd.read_csv("Advertising.csv")

Xs = df.loc[:, features].values
Ys = df.loc[:, ['Sales']].values

from sklearn.preprocessing import StandardScaler
Xs = StandardScaler().fit_transform(Xs)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(Xs)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['Sales']]], axis = 1)

columns = ['principal component 1', 'principal component 2']
X1 = finalDf.loc[:, columns].values
Y1 = finalDf.loc[:, 'Sales'].values.reshape(-1,1)

reg1 = LinearRegression()
reg1.fit(X1,Y1)

predictionData = np.array([[0.963044, 1.429895 ]])
reg1.predict(predictionData)

confidence = reg1.score(X1, Y1)
print('Confidence PCA:', confidence)