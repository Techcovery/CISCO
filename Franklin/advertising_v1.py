
import pandas as pd
#import numpy as np

#from sklearn import preprocessing, model_selection as cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

ad = pd.read_csv(r"C:\Users\fjesudha\Desktop\Advertising.csv", index_col=0)
sales = ad['Sales'].values.reshape(-1,1)
clt = LinearRegression()

for col in ad.columns[:-1]:
    print("\n\ncontribution through:", col)
    value = ad[col].values.reshape(-1,1)
    
    clt.fit(value, sales)
    predicted_sales = clt.predict(value)

    plt.scatter(value, sales)
    plt.plot(value, predicted_sales)
    plt.xlabel(col)
    plt.ylabel("Sales")
    plt.show()
    print("mse", mean_squared_error(sales, predicted_sales))
    print("r2", r2_score(sales, predicted_sales))







