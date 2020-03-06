import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import quandl as Quandl, math
from sklearn import preprocessing, model_selection as cross_validation, svm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Advertising.csv', index_col=0)
#print(df.head())

features = ['TV','Radio','Newspaper']
x = df.loc[:, features].values
y = df.loc[:,['Sales']].values
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pca = PCA()
principalComponents = pca.fit(x)

"""
#Checking the components
plt.figure()
plt.plot(np.cumsum(principalComponents.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Varaince %')
plt.title('Explained variance')
plt.show()
"""

#So select only one component for PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1','principal component 2'])
principalDf['Sales'] = y

#print (principalDf.isnull().any())

"""
principalDf.plot(x='principal component 1', y='principal component 2', style='o')
plt.title('PC1 vs Adj. Close') 
plt.xlabel('PC1') 
plt.ylabel('PC2') 
plt.show()
plt.figure(figsize=(15,10))
plt.tight_layout()
sns.distplot(principalDf['Sales'])
plt.show()
"""
X = principalDf.drop(['Sales'], axis=1)
y = principalDf['Sales']

#print (X)
#print (y)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

print ('***********************************PCA********************************************')
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
train_confidence = clf.score(X_train, y_train)
print('linear train Confidence:', train_confidence)
test_confidence = clf.score(X_test, y_test)
print('linear test Confidence:', test_confidence)
print('linear r coeff : %s' % clf.coef_)

y_orig = y_test
y_pred = clf.predict(X_test)

print('Linear Mean Absolute Error:', metrics.mean_absolute_error(y_orig, y_pred))
print('Linear Mean Squared Error:', metrics.mean_squared_error(y_orig, y_pred))
print('Linear Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_orig, y_pred)))

#print (df.describe())
#print (df[['TV','Radio','Newspaper']].corr())

"""
sns.pairplot(df, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', size=7, aspect=0.7)
plt.show()
"""
print ('*******************************Linear*****************************************************')
#print (df.isnull().values.any()) 

X = df.drop(['Sales'], axis=1)
y = df['Sales']

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
train_confidence = clf.score(X_train, y_train)
print('linear train Confidence:', train_confidence)
test_confidence = clf.score(X_test, y_test)
print('linear test Confidence:', test_confidence)
print('linear r coeff : %s' % clf.coef_)

y_pred = clf.predict(X_test)
y_orig = y_test

print('Linear Mean Absolute Error:', metrics.mean_absolute_error(y_orig, y_pred))
print('Linear Mean Squared Error:', metrics.mean_squared_error(y_orig, y_pred))
print('Linear Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_orig, y_pred)))

print ('**********************************Ridge*********************************************')
rr = Ridge(alpha=(10))
rr.fit(X_train, y_train)
r_test = rr.score(X_test, y_test)
r_train = rr.score(X_train, y_train)
print('train ridge confidence:', r_train)
print('test ridge confidence:', r_test)
print('ridge r coeff : %s' % rr.coef_)

y_pred = rr.predict(X_test)

print('Ridge Mean Absolute Error:', metrics.mean_absolute_error(y_orig, y_pred))
print('Ridge Mean Squared Error:', metrics.mean_squared_error(y_orig, y_pred))
print('Ridge Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_orig, y_pred)))

print ('\nMulti Linear Model behaves better than everything and the highest contributor to Sales is Radio !\n') 
