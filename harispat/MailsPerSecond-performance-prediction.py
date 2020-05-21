import pandas as pd
from sklearn.decomposition import PCA
from sklearn import model_selection
from sklearn import preprocessing, model_selection as cross_validation, neighbors, svm
from sklearn import preprocessing
import re

#########
df = pd.read_excel('/Users/harish/DSTraining/MPS.xlsx')
#########

df["Test Description"] = [x.split('+') for x in df["Test Description"]]
df["Test Description"] = list("".join(x) for x in df["Test Description"])
df["Version"] = [x.split('-') for x in df["Version"]]
df["Version"] = list(".".join(x[:2]) for x in df["Version"])
df['Models'] = df['Models'].map(lambda x: re.sub("C|V|X",'',x))
df = df.drop("Test Name",axis=1)

le = preprocessing.LabelEncoder()
le.fit(df["Test Description"])
df["Test Description"] = le.transform(df["Test Description"])
df["MPS"] = df["MPS"].astype(int)
print(df.tail(100))

Y = df["MPS"]
X = df.drop("MPS",axis=1)

lab_enc = preprocessing.LabelEncoder()
training_scores_encoded = lab_enc.fit_transform(Y)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, training_scores_encoded, test_size=0.1)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, min_samples_leaf=8, random_state=1)
model.fit(X_train, y_train)
predictions_rf = model.predict(X_test)
print("predictions MPS",predictions_rf)
score_mps = model.score(X_test,y_test)
print("score MPS",score_mps)
print("mps",y_test)

import matplotlib.pyplot as plt
plt.style.use('dark_background')
plt.plot(y_test, 'r-o', label = 'MPS value')
plt.plot(predictions_rf, 'y-o', label = 'predictions_MPS Score')
plt.vlines(x = 1,  ymin= 0, ymax = 1000, colors = 'g', linestyles='--')
plt.hlines(y = 749.367, xmin = 0, xmax = 6, linestyles= '--', colors= 'b')

plt.legend()
plt.title('ELBOW curve - predictions MPS and real MPS', fontweight='bold')

import numpy as np
n_groups = len(y_test)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.50
opacity = 0.8

rects1 = plt.bar(index, predictions_rf, bar_width,
alpha=opacity,
color='b',
label='predictions')

rects2 = plt.bar(index + bar_width, y_test, bar_width,
alpha=opacity,
color='g',
label='MPS')

plt.xlabel('Time')
plt.ylabel('MPS')
plt.title('MPS')
#plt.xticks(index + bar_width, ('A', 'B', 'C', 'D'))
plt.legend()

plt.tight_layout()
plt.show()
