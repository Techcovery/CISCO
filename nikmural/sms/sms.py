#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 13:59:22 2020

@author: nikmural
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import svm
import xgboost as xgb



df = pd.read_csv("spam.csv", encoding='latin-1')

df.head
df.columns


#remove unnamed columns, too many NaNs
df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
df = df.rename(columns = {'v1':'label','v2':'message'})
df.describe()


#The message of SMS has a correlation with spam. 
#More the length of msg, higher is the probability of it being a spam.
df['length'] = df['message'].apply(len)
df.head

df.hist(column='length',by='label',bins=60,figsize=(12,4));
plt.xlim(-40,950);


def text_process(text):
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    
    return " ".join(text)

#Process text data, remove stop words
sms_text = df['message'].copy()
sms_text = sms_text.apply(text_process)
vectorizer = TfidfVectorizer("english")
features = vectorizer.fit_transform(sms_text)

x_train, x_test, y_train, y_test = train_test_split(features, df['label'], test_size=0.3, random_state=111)

#Trying these 10 models
model1 = LogisticRegression(random_state=1, solver='lbfgs',max_iter=7600)
model2 = DecisionTreeClassifier(random_state=1)
model3 = KNeighborsClassifier()
model4 = VotingClassifier(estimators=[('lr', model1), ('dt', model2), ('kn', model3)], voting='hard')
model5 = BaggingClassifier(DecisionTreeClassifier(random_state=1))
model6 = RandomForestClassifier(random_state=1, n_estimators=100)
model7 = AdaBoostClassifier(random_state=1)
model8 = GradientBoostingClassifier(learning_rate=0.01,random_state=1)
model9 = xgb.XGBClassifier()
model10 = svm.SVC(gamma='scale')


models_list = [(model1, 'Logistic Regression'),
                (model2, 'Decision Tree'),
                (model3, 'KNeighbours Classifier'),
                (model4, 'Voting Classifier'),
                (model5, 'Bagging Classifier'),
                (model6, 'Random Forest Classifier'),
                (model7, 'ADA Boost Classifier'),
                (model8, 'Gradient Boost Classifier'),
                (model9, 'XGB Classifier'),
                (model10, 'SVM'),]

pred_scores = []
for m in models_list:
    model = m[0]
    model.fit(x_train, y_train)
    score = model.score(x_test,y_test)
    print(m[1], 'score:', score)
    pred_scores.append((m[1], [score]))
    
df1 = pd.DataFrame.from_items(pred_scores, orient='index', columns=['Score'])
df1

#Plot the scores
df1.plot(kind='bar', ylim=(0.8,1.0), figsize=(11,6), align='center', colormap="Accent")
plt.xticks(np.arange(10), df1.index)
plt.ylabel('Accuracy Score')
plt.title('Distribution by Classifier')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)