import unittest
import math
import numpy as np

from sklearn import preprocessing, model_selection as cross_validation, svm
from stock_predict import get_data, preprocess_data, pred_split, split
df = get_data()
df_test=df.head(100)
forecast_col = 'Adj. Close'
forecast_out = int(math.ceil(0.01 * len(df)))
df_test['label'] = df_test[forecast_col].shift(-forecast_out)
X = np.array(df_test.drop(['label'], 1))
X_testtrain= X[:-forecast_out]
y = np.array(df_test['label'])
y_testtrain = y[:-forecast_out]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_testtrain, y_testtrain, test_size=0.2)
X_train, X_test, y_train, y_test = split(X_testtrain,y_testtrain)
print(X_train,  y_train)
print(X_test,y_test)
