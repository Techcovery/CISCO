# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 18:25:24 2020

@author: fjesudha
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import PCA
from sklearn import model_selection as cross_validation, svm
# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier

df_spam = pd.read_csv('employee_file_spam.csv',  index_col=0, engine='python', names = [ "message", "label"])
df_spam.dropna(inplace=True)

df_ham = pd.read_csv('employee_file_ham.csv', index_col = 0, engine='python', names = ["message", "label"])
df_ham.dropna(inplace=True)

frames = [df_spam, df_ham]
emails = pd.concat(frames)

stopwords = ENGLISH_STOP_WORDS.union(['subject', 'recipient'])
vect = TfidfVectorizer(analyzer='word', stop_words=stopwords, max_df=0.3, min_df=2)

X = vect.fit_transform(emails.message)
features = vect.get_feature_names()

X_dense = X.todense()
print (X_dense.shape)
print (X.toarray())

y = emails['label']
y = y.apply({'ham':0, 'spam':1}.get)
#y.replace()
print(y)


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_dense, y, test_size=0.2)

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional

max_features = len(features)
model = Sequential()
#model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Embedding(max_features, 128))
model.add(Bidirectional(LSTM(64)))
#model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
model.fit(X_train, y_train,
          batch_size=100,
          epochs=4,
          validation_data=[X_test, y_test])

Y_predict = model.predict(X_test)
'''

pca = PCA(n_components=20).fit(X_dense)
coords = pca.transform(X_dense)


clf = svm.SVC(gamma='scale')
#Run SVM and calculate confidence
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print("confidence with svm is ", confidence)



# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_jobs=2, random_state=0,n_estimators=100)


# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
clf.fit(X_train, y_train)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=100, n_jobs=2, oob_score=False, random_state=0,
            verbose=0, warm_start=False)


# Apply the Classifier we trained to the test data (which, remember, it has never seen before)
print('Confidence with RFC is ',clf.score(X_test, y_test))

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)

print("Confidence with DTC is ", tree.score(X_test, y_test))

'''