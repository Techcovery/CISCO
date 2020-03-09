#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:54:06 2020

@author: nikmural
"""




import os, sys, email
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
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


#This dataset is taken from http://www2.aueb.gr/users/ion/data/enron-spam/ where
#it has already been classified as spam and ham. We traverse over this folder to
#generate the xl which has email name (folder name), content and the label
#ham or spam.
def create_excel():
    
    def progress(i, end_val, bar_length=50):
        '''
        Print a progress bar of the form: Percent: [#####      ]
        i is the current progress value expected in a range [0..end_val]
        bar_length is the width of the progress bar on the screen.
        '''
        percent = float(i) / end_val
        hashes = '#' * int(round(percent * bar_length))
        spaces = ' ' * (bar_length - len(hashes))
        sys.stdout.write("\rPercent: [{0}] {1}%".format(hashes + spaces, int(round(percent * 100))))
        sys.stdout.flush()
    
    
    HAM = 'ham'
    SPAM = 'spam'
    
    SOURCES = [
        ('data/beck-s',      HAM),
        ('data/farmer-d',    HAM),
        ('data/kaminski-v',  HAM),
        ('data/kitchen-l',   HAM),
        ('data/lokay-m',     HAM),
        ('data/williams-w3', HAM),
        ('data/BG',          SPAM),
        ('data/GP',          SPAM),
        ('data/SH',          SPAM)
    ]
    
    SKIP_FILES = {}

    def read_files(path):
        '''
        Generator of pairs (filename, filecontent)
        for all files below path whose name is not in SKIP_FILES.
        The content of the file is of the form:
            header....
            <emptyline>
            body...
        This skips the headers and returns body only.
        '''
        for root, dir_names, file_names in os.walk(path):
            for path in dir_names:
                read_files(os.path.join(root, path))
            for file_name in file_names:
                if file_name not in SKIP_FILES:
                    file_path = os.path.join(root, file_name)
                    if os.path.isfile(file_path):
                        past_header, lines = False, []
                        f = open(file_path, "r", encoding="latin-1")
                        contents = f.read()
                        contents = contents.replace('"','')
                        f.close()

                        yield file_path, contents

    def build_data_frame(l, path, classification):
        rows = []
        index = []
        for i, (file_name, text) in enumerate(read_files(path)):
            if ((i+l) % 100 == 0):
                progress(i+l, 52077, 50)
            rows.append({'message': text, 'class': classification})
            index.append(file_name)
       
        data_frame = DataFrame(rows, index=index)
        return data_frame, len(rows)

    def load_data():
        data = DataFrame({'message': [], 'class': []})
        l = 0
        for path, classification in SOURCES:
            data_frame, nrows = build_data_frame(l, path, classification)
            data = data.append(data_frame)
            l += nrows
    #     data = data.reindex(numpy.random.permutation(data.index))
        return data

    # This should take about 1 minute
    data=load_data()
    
    new_msg = []
    for msg in data.message:
       new_msg.append(msg.encode('utf-8').decode('utf-8'))
       
    data.loc[:, 'message'] = new_msg 
    #The email.xlsx thus generated is being used for further classification.
    data.to_excel("email.xlsx")

def score_all_models(x_train, x_test, y_train, y_test):
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
    
    model1.fit(x_train,y_train)

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
    return df1

def plot(df):
    #Plot the scores
    df.plot(kind='bar', ylim=(0.5,1.0), figsize=(11,6), align='center', colormap="Accent")
    plt.xticks(np.arange(10), df.index)
    plt.ylabel('Accuracy Score')
    plt.title('Distribution by Classifier')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  
def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df

def process_data():
    df = pd.read_excel('email.xlsx')
    emails_df = df.sample(n = 500)
    #emails_df.head
    emails_df.dropna()
    emails_df.rename(columns={ df.columns[0]: "file" }, inplace = True)
    
    
    ## Helper functions
    def get_text_from_email(msg):
        '''To get the content from email objects'''
        parts = []
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                parts.append( part.get_payload() )
        return ''.join(parts)

    def split_email_addresses(line):
        '''To separate multiple email addresses'''
        if line:
            addrs = line.split(',')
            addrs = frozenset(map(lambda x: x.strip(), addrs))
        else:
            addrs = None
        return addrs
        
    messages = list(map(email.message_from_string, emails_df['message']))

    # # Get fields from parsed email objects
    keys = messages[0].keys()
    # print(keys)
    for key in keys:
        emails_df[key] = [doc[key] for doc in messages]
    # Parse content from emails
    emails_df['content'] = list(map(get_text_from_email, messages))
    # Split multiple email addresses
    emails_df['From'] = emails_df['From'].map(split_email_addresses)
    emails_df['To'] = emails_df['To'].map(split_email_addresses)
    
    # # Extract the root of 'file' as 'user'
    emails_df['user'] = emails_df['file'].map(lambda x:x.split('/')[1])
    del messages
    
    print(emails_df.head())
    emails_df.drop('message', axis=1, inplace=True)
    print(emails_df.columns)
    
    
    def text_process(text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
        return " ".join(text)

    #Process text data, remove stop words
    email_text = emails_df['content'].copy()
    email_text = email_text.apply(text_process)
    vectorizer = TfidfVectorizer("english")
    cf = vectorizer.fit_transform(email_text)
    

    #Case 1: Use only email contents to identify spams/hams
    X = cf
    y = np.array(emails_df['class'])
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=111)
    df1 = score_all_models(x_train, x_test, y_train, y_test)
    print(df1)
    plot(df1)
    
    
    #Case 2: We will include length of email and check if it improves score.
    emails_df['length'] = emails_df['content'].apply(len)
    lf = emails_df['length'].as_matrix()
    X1 = np.hstack((X.todense(),lf[:, None]))
    x_train, x_test, y_train, y_test = train_test_split(X1, y, test_size=0.3, random_state=111)
    df2 = score_all_models(x_train, x_test, y_train, y_test)
    print(df2)
    plot(df2)
    
    
    #Case 3: we will try correlation matrix with other features and see if we can improve accuracy
    add_df = emails_df.drop(['content', 'user', 'file'], 1)
    num_df = handle_non_numerical_data(add_df)
    
    corr_matrix = num_df.corr()
    cor_target = abs(corr_matrix['class'])
    relevant_features = cor_target[cor_target>0.1]
    print(relevant_features)
    
    relevant_features = relevant_features.keys()
    X2 = num_df.loc[:, relevant_features].values
    X2 = StandardScaler().fit_transform(X2)
    
    X3 = np.hstack((X.todense(), X2, lf[:, None]))
    y = np.array(emails_df['class'])
    
    x_train, x_test, y_train, y_test = train_test_split(X3, y, test_size=0.3, random_state=111)
    df3 = score_all_models(x_train, x_test, y_train, y_test)
    print(df3)
    plot(df3)
    
    
    df = pd.concat([df1, df2, df3], axis=1)
    print(df)
    plot(df)
    
    
    
if __name__ == "__main__":
    process_data()