# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 18:25:24 2020

@author: fjesudha
"""

#Categorizing emails based on the message body. 

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def parse_raw_message(raw_message):
    lines = raw_message.split('\n')
    email = {}
    message = ''
    keys_to_extract = ['from', 'to']
    for line in lines:
        if ':' not in line:
            message += line.strip()
            email['body'] = message
        else:
            pairs = line.split(':')
            key = pairs[0].lower()
            val = pairs[1].strip()
            if key in keys_to_extract:
                email[key] = val
    return email

def parse_into_emails(messages):
    emails = [parse_raw_message(message) for message in messages]
    return {
        'body': map_to_list(emails, 'body'), 
        'to': map_to_list(emails, 'to'), 
        'from_': map_to_list(emails, 'from')
    }

def map_to_list(emails, key):
    results = []
    for email in emails:
        if key not in email:
            results.append('')
        else:
            results.append(email[key])
    return results


#Read the file as data frame
emails = pd.read_csv('emails.csv')

#Parse the message body into to, from and message data frame
email_df = pd.DataFrame(parse_into_emails(emails.message))

#drop the empty entries
email_df.drop(email_df.query("body == '' | to == '' | from_ == ''").index, inplace=True)

vect = TfidfVectorizer(analyzer='word', max_df=0.5, min_df=2)


X = vect.fit_transform(email_df.body)
features = vect.get_feature_names()

clf = KMeans(n_clusters=2,
            max_iter=100, 
            init='k-means++', 
            n_init=1)
labels = clf.fit_predict(X)


X_dense = X.todense()
pca = PCA(n_components=2).fit(X_dense)
coords = pca.transform(X_dense)

# Lets plot it again, but this time we add some color to it.
# This array needs to be at least the length of the n_clusters.
label_colors = ["#2AB0E9", "#2BAF74", "#D7665E", "#CCCCCC", 
                "#D2CA0D", "#522A64", "#A3DB05", "#FC6514"]
colors = [label_colors[i] for i in labels]

plt.scatter(coords[:, 0], coords[:, 1], c=colors)
# Plot the cluster centers

centroids = clf.cluster_centers_
centroid_coords = pca.transform(centroids)
plt.scatter(centroid_coords[:, 0], centroid_coords[:, 1], marker='X', s=200, linewidths=2, c='#444d60')
plt.show()
