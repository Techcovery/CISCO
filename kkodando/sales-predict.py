# Determine which add channel gives more sales revenue?

import pandas as pd
from sklearn import preprocessing, model_selection as cross_validation
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = pd.read_csv("Advertising.csv", index_col=0)
df.plot()

plt.show()

# Replace any NAN with mode
for column in df.columns:
    df[column].fillna(df[column].mode()[0], inplace=True)

# Check for linearity between TV Channel vs Sales Revenue
# We clearly see the linearity
plt.scatter(df['TV'], df['Sales'], color='red')
plt.title('TV Channel Vs Sales Revenue', fontsize=14)
plt.xlabel('TV Channel', fontsize=14)
plt.ylabel('Sales Revenue', fontsize=14)
plt.grid(True)
plt.show()

# Check for linearity between Radio Channel vs Sales Revenue
# We see some amount of linearity; but not as great as TV channel
plt.scatter(df['Radio'], df['Sales'], color='blue')
plt.title('Radio Channel Vs Sales Revenue', fontsize=14)
plt.xlabel('Radio Channel', fontsize=14)
plt.ylabel('Sales Revenue', fontsize=14)
plt.grid(True)
plt.show()

# Check for linearity between Newspaper vs Sales Revenue
# We see some amount of -ve linearity; but not as great as TV channel
plt.scatter(df['Newspaper'], df['Sales'], color='green')
plt.title('Newspaper Channel Vs Sales Revenue', fontsize=14)
plt.xlabel('Newspaper Channel', fontsize=14)
plt.ylabel('Sales Revenue', fontsize=14)
plt.grid(True)
plt.show()

y_label = 'Sales'
Y = df[y_label]

featureSet = ['TV', 'Radio', 'Newspaper']
X = df[featureSet]

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

lRegr = LinearRegression()
lRegr.fit(X_train, Y_train)
lRegrconf = lRegr.score(X_test, Y_test)
print('Linear Regression Confidence -->: ', lRegrconf)
print('Linear Regression Intercept --->:', lRegr.intercept_)
print('Linear Regression Coefficients --->:', lRegr.coef_)

polyRegr = PolynomialFeatures(degree=2)
X_train_ = polyRegr.fit_transform(X_train)
lRegr = LinearRegression()
lRegr.fit(X_train_, Y_train)

X_test_ = polyRegr.fit_transform(X_test)
confidence = lRegr.score(X_test_, Y_test)
print('Polynomial Regression with degree 2 confidence ----> :', confidence)

polyRegr = PolynomialFeatures(degree=3)
X_train_ = polyRegr.fit_transform(X_train)
lRegr = LinearRegression()
lRegr.fit(X_train_, Y_train)
# print(poly.get_feature_names())
X_test_ = polyRegr.fit_transform(X_test)
confidence = lRegr.score(X_test_, Y_test)
print('Polynomial Regression with degree 3 confidence ----> :', confidence)

# If we take just TV & Radio we get better confidence & thus better prediction.