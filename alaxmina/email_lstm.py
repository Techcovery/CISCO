from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb
import pandas as pd
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


max_features = 20000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 100
batch_size = 32

print('Loading data...')
emails = pd.read_csv('emails1.csv')
print emails.shape # (10000, 3)

def convert_emails(messages):
    for message in messages:
        lines = messages.split('\n')
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

    return {
        'body': map_to_list(emails, 'body'), 
        'to': map_to_list(emails, 'to'), 
        'from_': map_to_list(emails, 'from')
    }

email_df = pd.DataFrame(convert_emails(emails.message))
X = email_df['content']
Y = email_df['class']

#print(X)
#print(Y)
#Change ham and spam to one hot encoding
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1,1)

#split data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=4,
          validation_data=[x_test, y_test])


test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

accr = model.evaluate(test_sequences_matrix,y_test)
