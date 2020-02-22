from sklearn.decomposition import PCA
import quandl as Quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection as cross_validation, svm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
import seaborn as seabornInstance 
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures

style.use('ggplot')

df = Quandl.get("WIKI/GOOG",api_key='tJbUksb_TeZXtwAgDeia')
from sklearn.preprocessing import StandardScaler

features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Ex-Dividend', 'Split Ratio', 'Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']
x = df.loc[:, features].values
y = df.loc[:,['Adj. Close']].values
x = StandardScaler().fit_transform(x)
#print(x)

#find the variation
from sklearn.decomposition import PCA
pca = PCA()
principalComponents = pca.fit(x)

"""
# Checking the components
plt.figure()
plt.plot(np.cumsum(principalComponents.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Varaince %')
plt.title('Explained variance')
plt.show()
"""
# Shows variance is dependent only on one component

#So select only one component for PCA
pca = PCA(n_components=1)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1'])
principalDf['Adj. Close'] = y

#print (principalDf.isnull().any())

"""
principalDf.plot(x='principal component 1', y='Adj. Close', style='o')
plt.title('PC1 vs Adj. Close')  
plt.xlabel('PC1')  
plt.ylabel('Adj Close')  
plt.show()
plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(principalDf['Adj. Close'])
plt.show()
"""

"""
#Check Variation of Adj.Close wrt PC1
plt.figure()
plt.xlabel('Component 1')
plt.ylabel('Adj Close')
plt.title('1 Component PCA')
plt.scatter(principalDf['principal component 1'], y, c = 'blue', s = 50)
plt.show()
"""

forecast_col = 'Adj. Close'
forecast_out = int(math.ceil(0.01 * len(principalDf)))
principalDf['label'] = principalDf[forecast_col].shift(-forecast_out)
X = np.array(principalDf.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]

X = X[:-forecast_out]
principalDf.dropna(inplace=True)
y = np.array(principalDf['label'])
y_orig = y[-forecast_out:]

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)


print ('***********************************PCA********************************************')
#Try Linear Regression ?
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
train_confidence = clf.score(X_train, y_train)
print('PCA linear train Confidence:', train_confidence)
test_confidence = clf.score(X_test, y_test)
print('PCA linear test Confidence:', test_confidence)
print('PCA linear r coeff : %s' % clf.coef_)

y_pred = clf.predict(X_lately)

print('PCA Linear Mean Absolute Error:', metrics.mean_absolute_error(y_orig, y_pred))
print('PCA Linear Mean Squared Error:', metrics.mean_squared_error(y_orig, y_pred))
print('PCA Linear Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_orig, y_pred)))

# Try Poly Regression ?
poly = PolynomialFeatures(degree=2)
X_train_ = poly.fit_transform(X_train)
pf = LinearRegression()
pf.fit(X_train_, y_train)
X_test_ = poly.fit_transform(X_test)
p_test = pf.score(X_test_, y_test)
p_train = pf.score(X_train_, y_train)
print('Poly train poly confidence:', p_train)
print('Poly test poly confidence:', p_test)
print('Poly r coeff : %s' % pf.coef_)

y_pred = pf.predict(poly.fit_transform(X_lately))

print('Poly Mean Absolute Error:', metrics.mean_absolute_error(y_orig, y_pred))
print('Poly Mean Squared Error:', metrics.mean_squared_error(y_orig, y_pred))
print('Poly Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_orig, y_pred)))


# Linear Regression
print ('*******************************Linear*****************************************************')

df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
"""
seabornInstance.pairplot(df, x_vars=['Adj. Open',  'Adj. High',  'Adj. Low', 'Adj. Volume'], y_vars='Adj. Close', size=7, aspect=0.7)
plt.show()
"""

forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
df.dropna(inplace=True)
y = np.array(df['label'])
y_orig = y[-forecast_out:]

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
train_confidence = clf.score(X_train, y_train)
print('linear train Confidence:', train_confidence)
test_confidence = clf.score(X_test, y_test)
print('linear test Confidence:', test_confidence)
print('linear r coeff : %s' % clf.coef_)

y_pred = clf.predict(X_lately)

print('Linear Mean Absolute Error:', metrics.mean_absolute_error(y_orig, y_pred))
print('Linear Mean Squared Error:', metrics.mean_squared_error(y_orig, y_pred))
print('Linear Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_orig, y_pred)))

print ('**********************************Ridge*********************************************')
rr = Ridge(alpha=(1))
rr.fit(X_train, y_train)
r_test = rr.score(X_test, y_test)
r_train = rr.score(X_train, y_train)
print('train ridge confidence:', r_train)
print('test ridge confidence:', r_test)
print('ridge r coeff : %s' % rr.coef_)

y_pred = rr.predict(X_lately)

print('Ridge Mean Absolute Error:', metrics.mean_absolute_error(y_orig, y_pred))
print('Ridge Mean Squared Error:', metrics.mean_squared_error(y_orig, y_pred))
print('Ridge Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_orig, y_pred)))


print ('************************************Lasso*******************************************')
la = Lasso(alpha=(1))
la.fit(X_train,y_train)
a_test = la.score(X_test, y_test)
a_train = la.score(X_train, y_train)
print('train lasso confidence:', a_train)
print('test lasso confidence:', a_test)
print('lasso r coeff : %s' % la.coef_)

y_pred = la.predict(X_lately)

print('Lasso Mean Absolute Error:', metrics.mean_absolute_error(y_orig, y_pred))
print('Lasso Mean Squared Error:', metrics.mean_squared_error(y_orig, y_pred))
print('Lasso Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_orig, y_pred)))

"""
dlf = pd.DataFrame({'Actual': y_orig.flatten(), 'Predicted': y_pred.flatten()})
#print(dlf)
df1 = dlf.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

plt.scatter(X_test[:,1], y_test,  color='gray')
plt.plot(X_lately[:,1], y_pred, color='red', linewidth=2)
plt.show()
"""

print ("\nGoing by Root Mean Squared Error, PCA with Linear Regression seems to be the best since it's the least one.\n")
