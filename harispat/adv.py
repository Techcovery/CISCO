import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import preprocessing
import statsmodels.api as smc
from sklearn import model_selection as cross_validation
import quandl as Quandl, math
from sklearn.preprocessing import PolynomialFeatures

X_ = [[130,20,1]]

df=pd.read_csv('Advertising.csv')
df.head()
X=df.drop(['Sales','Unnamed: 0'],axis=1)
print(X)
y = df['Sales']
y.head()


#reg = LinearRegression()
#reg.fit(X,y)
#print(reg.coef_)
#print(reg.intercept_)

#prepare the test and train data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
#set polynomial to degree 3
poly = PolynomialFeatures(degree=3)
#prepare data for degree 3
X_train_ = poly.fit_transform(X_train)
X_test_ = poly.fit_transform(X_test)
X_= poly.fit_transform(X_)

#fit the degree 3 data in linear regression
lg = LinearRegression()
lg.fit(X_train_,y_train)
print("The linear model is: Y = {:.3} + {:.3}*TV + {:.3}*radio + {:.3}*newspaper".format(reg.intercept_, reg.coef_[0], reg.coef_[1], reg.coef_[2]))

print(lg.score(X_test_,y_test))
p = lg.predict(X_)
print("predicted sale for is %s"%format(p))