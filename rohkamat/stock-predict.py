
# pip install quandl

import quandl as Quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection as cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime




style.use('ggplot')

#Use Pandas to populate the data frame
df = Quandl.get("WIKI/AMZN",api_key='tJbUksb_TeZXtwAgDeia')

print('All Features in Data Sample \n')
for col in df.columns: 
    print(col) 
    
#print(df.head(5))
'''
# Create correlation matrix
corr_matrix = df.corr().abs()
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.99)]
print ("To Drop:", to_drop)
'''
# ignoreing open close high and low and volume, since we have adjusted versions of these
# Stock split has actually happened 3 times in 1998,1999 - we shouldnt ignore this

# Ex-Dividend is the amount by which the next day's price will be reduced when dividend is paid.
# Amazon doesnt pay dividends - so we can skip this.

# These are the features we have chosen   ---- RK  QNS : Can we choose others?
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]

print('\nInitial Input Features with Data Sample \n')
print(df.head(5))

#RK - Instead replace with the mod of the other values of that feature.
# and do this before using the values in reduction.

#Confidence before 0.9790714456178649  
#Confidence after: 0.9794053827070448
'''
df.fillna(df['Adj. Open'].mode()[0], inplace=True)
df.fillna(df['Adj. High'].mode()[0], inplace=True)
df.fillna(df['Adj. Low'].mode()[0], inplace=True)
df.fillna(df['Adj. Close'].mode()[0], inplace=True)
df.fillna(df['Adj. Volume'].mode()[0], inplace=True)
'''

# replace with Mean
# before Confidence: 0.9781616886507051
# After Confidence: 0.980721784221246
df.fillna(df['Adj. Open'].mean(), inplace=True)
df.fillna(df['Adj. High'].mean(), inplace=True)
df.fillna(df['Adj. Low'].mean(), inplace=True)
df.fillna(df['Adj. Close'].mean(), inplace=True)
df.fillna(df['Adj. Volume'].mean(), inplace=True)

# replace with Median ? doesnt seem right for this


# RK Can we reduce features in a better way?

# Variance Thresholds remove features which may not change much, thus not having impact
# Get co-relation between 2 related features - high low

H_L = df['Adj. High'].corr(df['Adj. Low'])

# Correlation : 0.9998661880929293
print("Correlation between Adj high and Adj Low : ",H_L)

# Get co-relation between 2 related features - open and close
O_C = df['Adj. Open'].corr(df['Adj. Close'])
# Correlation : 0.9998419767063804
print("Correlation between Adj Open and Adj Close : ",O_C)

# Reduce Features Adj. High and Adj. Low to One percentage HL_PCT
# Reduce Features Adj. Open and Adj. Close to another percentage called PCT_change

#Calculate percentages
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0


# Now our data frame has 
#Predict closing values
# df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# RK remove one each of the closely related features
'''
   Before:
Confidence: 0.9758470868117777
Empty DataFrame
Columns: [Adj. Close, HL_PCT, PCT_change, Adj. Volume, label, Forecast]
   After:    
Confidence: 0.9755350589131841
Empty DataFrame
Columns: [Adj. Close, Adj. High, Adj. Volume, label, Forecast]

df = df[['Adj. Close', 'Adj. High', 'Adj. Volume']]
'''


# Try using PCA

# PCA is effected by scale so first need to scale the features in the data set
from sklearn.preprocessing import StandardScaler

features = ['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']
std_x = df.loc[:, features].values
std_x = StandardScaler().fit_transform(std_x)

# After standardized
print(std_x[:5])

# Now apply PCA and get 3 components
from sklearn.decomposition import PCA

# With 3 Components 
# Confidence: 0.9739836768494355
# With 2 Components
# Confidence: 0.9773038795026069
# With 4 Components 
# Confidence: 0.9783824148662673
pca = PCA(n_components=4)
principalComponents = pca.fit_transform(std_x)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['PComp1', 'PComp2', 'PComp3','PComp4'])
'''
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['PComp1', 'PComp2'])
'''
print('\nAfter PCA Data Sample \n')
print(principalDf.head(5))





'''
print('\nAfter Feature Reduction Features Data Sample \n')
print(df.head(5))
'''



'''
# RK - Here we are replacing any NA values with some fixed number ---
#Replace any NAN so that data is not lost
# df.fillna(value=-99999, inplace=True)

# Before 0.976222016756481
# After 0.9807080055407879
df.fillna(df['HL_PCT'].mode()[0], inplace=True)
df.fillna(df['PCT_change'].mode()[0], inplace=True)
'''

#Before Confidence: 0.980721784221246
# After Confidence: 0.9785016781987287
# replace with Mean
'''
df.fillna(df['HL_PCT'].mean(), inplace=True)
df.fillna(df['PCT_change'].mean(), inplace=True)
'''



forecast_col = 'Adj. Close'
# RK -  Whats hapenning here?
forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

# X is the input data and excludes labels to be predicted
#  X = np.array(df.drop(['label'], 1))

# With PCs instead of features
X = np.array(principalDf)


#Preprocessing gets values between -1 and 1
#  X = preprocessing.scale(X)


#Input for which prediction is needed
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)


#Output to be predicted
y = np.array(df['label'])

#Use linear regression classifier
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print('Confidence:', confidence)

#Predict for the forecast set
forecast_set = clf.predict(X_lately)
df['Forecast'] = np.nan
#Predict for every day after the last day in training set
print(df.head(0))

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
    

