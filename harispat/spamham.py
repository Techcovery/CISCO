import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.model_selection import train_test_split


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
                    x += 1

            df[column] = list(map(convert_to_int, df[column]))

    return df


def genData():
    file = "Enron_Labelled_Spam_Ham.csv"
    df = pd.read_csv(file)

    print(df.columns)

    df['NumRecipients'] = df['To'].str.count(",") + 1
    df.fillna(df['NumRecipients'].mode()[0], inplace=True)
    df['NumRecipients'] = df['NumRecipients'].astype('int')

    # df['SubLen'] = df['Subject'].str.len()
    # df.fillna(df['SubLen'].mode()[0], inplace=True)
    # df['SubLen'] =  df['SubLen'].astype('int')
    # df['FromLen'] = df['From'].str.len()
    # df['FromLen'] = df['FromLen'].astype('int')

    df['IsSpam'] = df['IsSpam'].fillna(0)
    df['IsSpam'] = df['IsSpam'].astype('int')

    # df['contentLen'] = df['content'].str.len()
    # df.fillna(df['contentLen'].mode()[0], inplace=True)
    # df['contentLen'] = df['contentLen'].astype('int')
    # df1 = df[['NumRecipients', 'Subject', 'contentLen','FromLen','IsSpam',]]

    df1 = df[['NumRecipients', 'Subject', 'content', 'From', 'IsSpam' ]]
    df1 = handle_non_numerical_data(df1)
    return df1


def svm(X_train, X_test, y_train, y_test):
    from sklearn import svm
    svmModel = svm.SVC(gamma='scale')
    svmModel.fit(X_train, y_train)
    scoreSVM = svmModel.score(X_test, y_test)
    print('SVM Accuracy:', scoreSVM)


def logistic(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LogisticRegression
    LR = LogisticRegression(random_state=1, solver='lbfgs', max_iter=2500)
    LR.fit(X_train, y_train)
    score = LR.score(X_test, y_test)
    print('Logistic-Regression accuracy:', score)


if __name__ == "__main__":
    df = genData()
    X = df.drop(['IsSpam'], axis=1)
    Y = df['IsSpam']
    # X = preprocessing.scale(X)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2)
    svm(X_train, X_test, y_train, y_test)
    logistic(X_train, X_test, y_train, y_test)