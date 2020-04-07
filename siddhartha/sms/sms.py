# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
# import Natural Language toolkit
import nltk 
#import Beutiful Soup
from bs4 import BeautifulSoup
#import string for list of puntuations
import string
#import the stop word list
from nltk.corpus import stopwords
# import Tokenizer
from nltk.tokenize import RegexpTokenizer
# import Lemmatizer
from nltk.stem import WordNetLemmatizer
# Import stemmer
from nltk.stem.porter import PorterStemmer #Hanging
from nltk import stem

df = pd.read_csv('spam_encoded.csv', encoding='utf8')
df.head()
df.shape

# drop the columns other than the ham/spam and Message (keep v1, v2)
df.drop(df.columns[[2,3,4]], axis=1, inplace=True)
df2 = df[['spam', 'original_message']].copy()
# Rename v1, v2 as Label and Message and create another dataset
df2["spam"]= df2["spam"].replace(1, "spam")
df2["spam"]= df2["spam"].replace(0, "ham")

df2 = df2.rename(columns={"spam": "Label", "original_message": "Message"})

# cleaning unicode chars
def clean_text(row):
    # return the list of decoded cell in the Series instead 
    return [r.decode('unicode_escape').encode('ascii', 'ignore') for r in row]

#df2['Message'] = df2['Message'].values.apply(lambda x: x.decode('unicode_escape').encode('ascii', 'ignore').strip())

"""
we will use the following steps to get the data ready for NLP
    Remove HTML
    Tokenization + Remove punctuation
    Remove stop words
    Lemmatization or Stemming
    
    Goto the following link for more information:
        https://towardsdatascience.com/nlp-for-beginners-cleaning-preprocessing-text-data-ae8e306bef0f
"""
#Remove HTML
# NOt applicable for this dataset

# Remove punctuation
def remove_punctuation(text):
    no_punct = "".join([c for c in text if c not in string.punctuation])
    return no_punct

df2['Message'] = df2['Message'].apply(lambda x: remove_punctuation(x))
df2.head()

# Tokenize the Message so that it can be used in pattern recognition
# Instantiate Tokenizer

tokenizer = RegexpTokenizer(r'\w+')
df2['Message'] = df2['Message'].apply(lambda x: tokenizer.tokenize(x))
df2.head()

# Remove stop words
def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words('english')]
    return words
df2['Message'] = df2['Message'].apply(lambda x: remove_stopwords(x))
df2.head()

## To overcome the unicode related issue in stemmer and lemmetizer
import sys
# sys.setdefaultencoding() does not exist, here!
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')

# Stemming & Lemmatizing
#We will use Lemmatization here as it gives back the root word
lemmatizer = WordNetLemmatizer()
def word_lemmatizer(text):
    lem_text = [lemmatizer.lemmatize(i.encode('utf-8')) for i in text]
    return lem_text
df2['Message'] = df2['Message'].apply(lambda x: word_lemmatizer(x))

#Apply Stemmer
stemmer = stem.SnowballStemmer('english')
def word_stemmer(text):
    stem_text = "".join([stemmer.stem(i.encode('utf-8')) for i in text])
    return stem_text
#df2['Message'] = df2['Message'].apply(lambda x: word_stemmer(x))
df2.head()

# Training the model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df2['Message'], df2['Label'], test_size = 0.2, random_state = 1)

# training the vectorizer
def tokens(x):
    return x.split(',')

"""Note: Vectorizer does not take anything other than string and it
    also tries to convert the values in to lower case, hence need to use
    the options to eliminate that error
    """
    
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(lowercase=False)
X_train = vectorizer.fit_transform(str (item) for item in X_train)

# Building and Testing the Classifier
# Use 1: Support Vector Machine

from sklearn import svm
svm = svm.SVC(C=300)
svm.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix
X_test = vectorizer.transform(str(i) for i in X_test)
y_pred = svm.predict(X_test)
print(confusion_matrix(y_test, y_pred))

def pred(msg):
    msg = vectorizer.transform([msg])
    prediction = svm.predict(msg)
    return prediction[0]

pred('winner$$$$ replyyyyy "winnn" $$$$')

pred("WINNER!! As a valued network customer you have won $1000")


