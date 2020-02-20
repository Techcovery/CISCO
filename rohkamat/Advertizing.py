#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 22:14:09 2020

@author: rohkamat
"""

# Objective : How to spend the ad budget to get best sales increase?

import pandas as pd

# Read the data, drop the unnamed index col
df = pd.read_csv("/Users/rohkamat/Documents/MLTraining/math-for-mlv1-master/linear-regression/Advertising.csv",index_col=0)

print (df.head())

# Clean up the data, remove all NAs

df.dropna(inplace=True)

# find which one has most co-relation with Sales increase

# is it TV, Radio or news paper

# Radio to Sales
R_S = df['Radio'].corr(df['Sales'])
print("Correlation between Radio & Sales : ",R_S)

# TV to Sales
T_S = df['TV'].corr(df['Sales'])
print("Correlation between TV & Sales : ",T_S)

# NewsPaper to Sales
N_S = df['Newspaper'].corr(df['Sales'])
print("Correlation between Newspaper & Sales : ",N_S)


# Want to predict Sales given a distribution of the 3 ad spends

#Output to be predicted
y = np.array(df['Sales'])

# scale
# 3 features
# features = ['TV', 'Radio', 'Newspaper']

#Confidence: 0.8860762469403964
# 2 features - drop Newspaper
features = ['TV', 'Radio']
Std_X = df.loc[:, features].values
Std_X = StandardScaler().fit_transform(Std_X)



# How about giving more weightage to TV ?


# Now apply PCA and get 2 components
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(Std_X)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['PComp1', 'PComp2'])

X = np.array(principalDf)

# Split into Test and Training

#Use linear regression classifier
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)

# With 3 values and linear : Confidence: 0.881235757127077
# With 2 PComponents Confidence: 0.903227845163485

print('Linear Regression Confidence:', confidence)






