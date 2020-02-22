import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, model_selection as cross_validation, svm
csv_file = "/Users/rahulgupta/Git/machine_learning/math-for-mlv1/rahul/Advertising.csv"
df = pd.read_csv(csv_file)

data_top = df.head()

sales_map = {}
for col in list(df.columns):
    sales_map[col] = df[col]

media = ["TV", "Radio", "Newspaper"]
predicted_map ={}
for test in media:
    height = np.array (sales_map[test])
    weight = sales_map["Sales"]
    reg = LinearRegression()
    reg.fit(height.reshape(-1, 1), weight)
    # get linear regression coefficient

    m = reg.coef_[0]
    b = reg.intercept_
    print("slope=", m, "intercept=", b)
    # plot the data points and the line
    plt.scatter(height, weight, color="black")
    predicted_values = [reg.coef_ * i + reg.intercept_ for i in height]
    plt.plot(height, predicted_values, 'b')
    plt.xlabel("%s" %test)
    plt.ylabel("Sales")
    plt.show()
    print("Sales for expenditure in %s = 100 ->>>>>" %test, reg.predict(X=[[100]]))
    predicted_map["%s" %test] = reg.predict(X=[[100]])
    # Use linear regression classifier
    #X_train, X_test, y_train, y_test = cross_validation.train_test_split(height, weight, test_size=0.2)
    #print (y_train)
    #clf = LinearRegression(n_jobs=-1)
    #clf.fit(X_train, y_train.reshape(-1, 1))
    #confidence = clf.score(X_test, y_test)
    #print('Confidence:', confidence)


print ("Final predicted_map -> ")
print (predicted_map)

best_media = max(predicted_map, key=predicted_map.get)
print("Best Media: %s"%best_media)
