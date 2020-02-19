import quandl as Quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection as cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
import pickle

style.use('ggplot')

#Get function we did not pass any input
#get function gives output as data frame from Quandl

def get_data():
    df = Quandl.get("WIKI/AMZN",api_key='tJbUksb_TeZXtwAgDeia')
    return df
#df = get_data()

#Preprocess function takes input as df
#it clean the data and send output as cleaning data df_clean

def preprocess_data(df):
    #taking Adj. Open,  Adj. High,  Adj. Low,  Adj. Close, Adj. Volume from dataframe
    df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
    #calculating the high low percentage
    df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
    #calculating high low percentage
    df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
    #adding HL_PCT, PCT_change to the dataframe
    df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
    df.fillna(value=-99999, inplace=True)
    df.dropna(inplace=True)
    return df
#df = preprocess_data(df)
#print(df)

#pred_split function takes cleaning data(df_clean) as input
#it will split data as prediction and testtrain data
#it gives output as X_pred,y_pred,X_testtrain,y_testtrain
def pred_split(df):
    forecast_col = 'Adj. Close'
    #calculating the forecast_out
    forecast_out = int(math.ceil(0.01 * len(df)))
    df['label'] = df[forecast_col].shift(-forecast_out)
    X = np.array(df.drop(['label'], 1))
    #assign the X_testtrain data
    X_testtrain= X[:-forecast_out]
    #assign the X_pred data
    X_pred= X[-forecast_out:]
    y = np.array(df['label'])
    #assign the y_testtrain data
    y_testtrain = y[:-forecast_out]
    #assign the y_pred data
    y_pred = y[-forecast_out:]
    return X_pred,y_pred,X_testtrain,y_testtrain
#X_pred,y_pred,X_testtrain,y_testtrain = pred_split(df)
#print(X_pred,y_pred)
#print(X_testtrain[0],y_testtrain[0])

#split function takes X_testtrain,y_testtrain data as Input
#it gives output as X_train,X_test,y_train,y_test
def split(X_testtrain,y_testtrain):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_testtrain, y_testtrain, test_size=0.2)
    return X_train, X_test, y_train, y_test
#X_train, X_test, y_train, y_test = split(X_testtrain,y_testtrain)
#print(X_train[0])
#print(X_test[0])
#print(y_train[0])
#print(y_test[0])

#train function takes input as Taining data(X_train,y_train)
#it trains the  model
#it gives model as  output

def train(X_train,y_train):
    model = LinearRegression(n_jobs=-1)
    model.fit(X_train,y_train)
    m=model.coef_
    b=model.intercept_
    return m,b,model
#m,b,model = train(X_train,y_train)
#print("coeff,intercept",m,b)

#model = train(X_train,y_train)
#print(model)

#savemodel takes model as input
#it save the model in pickle file

def savemodel(model):
    with open('linearregression.pickle','wb') as f:
        pickle.dump(model, f)
#savemodel(model)

#test functions takes testing data(X_test,y_test)as input
#it gives confidence interval as output

def test(X_test,y_test,model):
    confidence = model.score(X_test, y_test)
    return confidence
#confidence = test(X_test,y_test)
#print('Confidence:', confidence)

#pred function takes input as prediction data(X_pred)
#it gives output as future predicted values

def predict(X_pred,model):
    #find the future prediction values
    y_predict = model.predict(X_pred)

    return y_predict
#y_predict=predict(X_pred)
#print(y_predict)

#visualixzation function takes closing date and forecast values as input
#it gives out as graph

def visualize(date,price,df,y_predict):
#Plot the graph for visualizayion
  df['dates'] = np.nan
    #Predict for every day after the last day in training set
    #print(df.head(0))
  last_date = df.iloc[-1].name
  last_unix = last_date.timestamp()
  one_day = 86400
  next_unix = last_unix + one_day
  for i in y_predict:
      next_date = datetime.datetime.fromtimestamp(next_unix)
      next_unix += 86400
      df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

  plt.plot(df[date])
  plt.plot(df[price])
  plt.legend(loc=4)
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.show()
#visualize('dates','Adj. Close')


def main():
    df = get_data()
    df = preprocess_data(df)
    X_pred,y_pred,X_testtrain,y_testtrain = pred_split(df)
    X_train, X_test, y_train, y_test = split(X_testtrain,y_testtrain)
    #model = train(X_train,y_train)
    m,b,model = train(X_train,y_train)
    print("coeff,intercept",m,b)
    savemodel(model)
    confidence = test(X_test,y_test,model)
    print('Confidence:', confidence)
    y_predict=predict(X_pred,model)
    print(y_predict)
    visualize('dates','Adj. Close',df,y_predict)

if __name__ == "__main__":
    main()
