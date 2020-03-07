#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:05:13 2020

@author: rohkamat
"""


import numpy as np
from sklearn import preprocessing,model_selection as cross_validation, neighbors
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

''' ----------------- UTILITY FUNCTIONS -------------------'''

def loadData():

    dataFile = '/Users/rohkamat/Documents/MLTraining/Enron_Rohini_Labelled_Spam_Ham.csv'

    # Load one csv, drop unamed col
    # Drop Unamed index
    df = pd.read_csv(dataFile,nrows=1500,index_col=0)

    print(df.head(5))
    
    return df



# Drop columns

def chooseRelevantColumns(df):
    '''
    # Message-ID
    df = df.drop(['Message-ID'], 1)
    # Date
    df = df.drop(['Date'], 1)
    # Folder name doesnt make much sense to keep
    df = df.drop(['X-Folder'], 1)
    # We dont need To, BCC and CC because we got new columns with counts
    df = df.drop(['To'], 1)
    df = df.drop(['X-cc'], 1)
    df = df.drop(['X-bcc'], 1)
    
    Instead of dropping so many, just choose the ones we want
    '''
    # IsSpam is the label I have filled in manually for 1500 rows
    
    # For now choosing only 4 features, dropping SubjectLen and From for now. IsSpam is the label
    
    df1 = df[['NumRecipients','MailTextLen','FromLen','isExternal','IsSpam']]
    
    return df1
    


#Preprocess - clean up data
def preProcessData(df):

    '''
    To -> how many receipents?
    There are duplicates in the To! First remove the duplicates then count
    Check - Do We need to repeat for cc, bcc ? --- No we dont! These are just x-cc, x-bcc - no value in these
    
    From -> i can check if internal or external
    I can also check the length of the from because spam-ish stuff has long sender names.
    
    '''
    print(df['To'].head(5))
    df['To'].unique()
    '''
    This is a strange way to get the count of the values. But it works.
    A better solution would be prefered
    '''
    df['NumRecipients'] = df['To'].str.count(",") + 1
    
    #Made a new column
    print(df['NumRecipients'].head(5))
    # Verify that i got it right
    # print ( df.loc[df['NumRecipients'] > 1 , 'To'] )
    
    # X-cc and X-BCC can be ignored since we dont have the real mail ids in CC and BCC
    
    ''' New Features I have prepared from the existing columns '''
    
    df['MailTextLen'] = df['content'].str.len()
    df['SubjectLen'] = df['Subject'].str.len()
    
    
    # isInternal gave negetive co-relation as expected, so flipped it and made it External for easy reading
    df['isExternal'] = ~df['From'].str.contains('@enron.com')
    
    df['isExternal'] = df['isExternal'].astype('int') 
    
    df['FromLen'] = df['From'].str.len()
    
    # Fill NAs with Mode
    df.fillna(df['NumRecipients'].mode()[0], inplace=True)
    df.fillna(df['MailTextLen'].mode()[0], inplace=True)
    df.fillna(df['SubjectLen'].mode()[0], inplace=True)
    
    ''' This is my manually filled label, fill 0 where not filled. 1 means isSpam '''
    df['IsSpam'] = df['IsSpam'].fillna(0)
    
    
    return df



def getPrintCorrelations(df):

    ''' ----------------- Co-Relations  -------------------'''

    NR_S = df['NumRecipients'].corr(df['IsSpam'])
    print("Correlation between NumRecipients & IsSpam : ",NR_S)

    MTL_S = df['MailTextLen'].corr(df['IsSpam'])
    print("Correlation between MailTextLen & IsSpam : ",MTL_S)

    E_S = df['isExternal'].corr(df['IsSpam'])
    print("Correlation between isExternal & IsSpam : ",E_S)

    FL_S = df['FromLen'].corr(df['IsSpam'])
    print("Correlation between FromLen & IsSpam : ",FL_S)

    '''
    Output
    Correlation between NumRecipients & IsSpam :  0.05535540763185523
    Correlation between MailTextLen & IsSpam :  0.0721955588392555
    Correlation between isExternal & IsSpam :  0.22198980686570582   <----
    Correlation between FromLen & IsSpam :  0.062191331241662755
    
    '''


'''
Uses only 2 impactful features
And returns the Test and Train split as 20:80
'''

def UseTwoFeatures(df):
    # We will use the 2 most impactful features from Co-relation: isExternal and MailTextLen
    Twofeatures = ['isExternal', 'MailTextLen']

    #Output to be predicted
    y = np.array(df['IsSpam'])
    
    Std_X = df.loc[:, Twofeatures].values
    Std_X = StandardScaler().fit_transform(Std_X)
    
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(Std_X, y, test_size=0.2)
    
    return  X_train, X_test, y_train, y_test
    



''' USE PCA to reduce to 3, and consider all 4 features this time 
    And returns the Test and Train split as 20:80
'''

def UsePCAFeatures(df):

    features = ['NumRecipients','MailTextLen','FromLen','isExternal']
    
    Std_X = df.loc[:, features].values
    Std_X = StandardScaler().fit_transform(Std_X)
    
    #Output to be predicted
    y = np.array(df['IsSpam'])
    
    # Now apply PCA and get 3 components
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(Std_X)
    
    principalDf = pd.DataFrame(data = principalComponents
                 , columns = ['PComp1', 'PComp2', 'PComp3'])
    
    X = np.array(principalDf)
    
    # Split into Test and Training
    
    #Use linear regression classifier
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
    return  X_train, X_test, y_train, y_test



''' Execute all Regression types and return their confidence '''

def executeRegressionTypes(X_train, X_test, y_train, y_test):
    print( '\n\n----------- Regression types -----------\n\n')
    
    ''' 1. Linear Regression '''
    
    #Use linear regression classifier
    
    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)
    confidenceLR = clf.score(X_test, y_test)
    
    print('Linear Regression Confidence:', confidenceLR)
    
    
    ''' 2. Ridge Regression '''
    
    # Ridge
    from sklearn.linear_model import Ridge
    
    alpha = 0.2  #An alpha of 1, will result in a model that acts identical to Linear Regression.
    
    rr = Ridge(alpha)
    rr.fit(X_train, y_train)
    confidenceR = rr.score(X_test, y_test)
    print('Ridge Regression Confidence:', confidenceR)
    
    
    ''' 3. Lasso Regression '''
    
    from sklearn.linear_model import Lasso
    
    lasso = Lasso()
    lasso.fit(X_train,y_train)
    confidenceL = lasso.score(X_test, y_test)
    print('Lasso Regression Confidence:', confidenceL)
    
    return confidenceLR, confidenceR, confidenceL
    
    # Polynomial Regression ????

    




''' -----------------  All types of Classification  -------------------'''

def executeClassificationTypes(X_train, X_test, y_train, y_test, printSuppress=False):
    
    if (printSuppress != True) :
        print( '\n\n----------- Classification types -----------\n\n')
    
    ''' 1. Nearest Neighbours Classifier '''
    
    #Nearest neighbor classifier from skikit learn
    
    #### Ideally this should be only used on small datasets - but here we only have 1500 rows so it should be fine 
    
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)
    
    accuracyKNN = clf.score(X_test, y_test)
    if (printSuppress != True) :
        print('Confidence with K-Nearest Neighbours Classifier :', accuracyKNN)
    
    
    ''' 2. DecisionTree  Classifier '''
    
    from sklearn.tree import DecisionTreeClassifier
    
    
    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X_train, y_train)
    accuracyDTTest = tree.score(X_test, y_test)
    if (printSuppress != True) :
        print('DecisionTree Accuracy on the training subset: {:.3f}'.format(tree.score(X_train, y_train)))
        print('DecisionTree Accuracy on the test subset: {:.3f}'.format(tree.score(X_test, y_test)))
        print('DecisionTree Accuracy Return', accuracyDTTest )
    
    return accuracyKNN, accuracyDTTest


''' 3. Ensembles '''

def executeEnsembleTypes(X_train, X_test, y_train, y_test, printSuppress=False):

    if (printSuppress != True) :
        print( '\n\n----------- Ensembles -----------\n\n')
    
    from sklearn.ensemble import BaggingClassifier
    model4 = BaggingClassifier(DecisionTreeClassifier(random_state=1))
    model4.fit(X_train, y_train)
    scoreB = model4.score(X_test,y_test)
    
    if (printSuppress != True) :
        print(' Bagging accuracy',scoreB )
    
    #Random bags and average voting is done in Random forest
    from sklearn.ensemble import RandomForestClassifier
    model5= RandomForestClassifier(random_state=1, n_estimators=100)
    model5.fit(X_train, y_train)
    scoreRF = model5.score(X_test,y_test)
    if (printSuppress != True) :
        print(' Random forest score:', scoreRF )
    
    
    
    model1 = LogisticRegression(random_state=1, solver='lbfgs',max_iter=7600)
    model2 = DecisionTreeClassifier(random_state=1)
    
    model1.fit(X_train,y_train)
    scoreLR = model1.score(X_test,y_test)
    if (printSuppress != True) :
        print(' Logistic Regression model 1 accuracy:', scoreLR)
    
    model2.fit(X_train,y_train)
    scoreDT = model2.score(X_test,y_test)
    if (printSuppress != True) :
        print(' DescisionTree model 2 accuracy:', scoreDT)
    
    model3 = KNeighborsClassifier(n_neighbors=5)
    model3.fit(X_train,y_train)
    scoreKNN = model3.score(X_test,y_test)
    if (printSuppress != True) :
        print(' KNN model 3 accuracy:',scoreKNN)
    
    
    from sklearn.ensemble import VotingClassifier
    
    modelv = VotingClassifier(estimators=[('lr', model1), ('dt', model2), ('kn',model3)], voting='hard')
    modelv.fit(X_train,y_train)
    scoreV_LR_DT_KN = modelv.score(X_test,y_test)
    if (printSuppress != True) :
        print(' LR, DT, KN   Ensemble model accuracy:',scoreV_LR_DT_KN)
    
    modelt = VotingClassifier(estimators=[('bc', model4), ('rf', model5)], voting='hard')
    modelt.fit(X_train,y_train)
    scoreV_BC_RF = modelt.score(X_test,y_test)
    if (printSuppress != True) :
        print(' BC, RF   Ensemble model accuracy:',scoreV_BC_RF)
    
    return scoreB, scoreRF, scoreLR, scoreDT, scoreKNN, scoreV_LR_DT_KN, scoreV_BC_RF
    
    
    
def executeBoostingTypes(X_train, X_test, y_train, y_test, printSuppress=False):   

    if (printSuppress != True) :
        print ('\n-------BOOSTING----------\n')
    
    #AdaBoost - Adaptive boosting
    from sklearn.ensemble import AdaBoostClassifier
    model6 = AdaBoostClassifier(random_state=1)
    model6.fit(X_train, y_train)
    scoreAB = model6.score(X_test,y_test)
    if (printSuppress != True) :
        print('Ada Boost accuracy',scoreAB)
    
    #Gradient Boosting
    from sklearn.ensemble import GradientBoostingClassifier
    model7= GradientBoostingClassifier(learning_rate=0.01,random_state=1)
    model7.fit(X_train, y_train)
    scoreGB = model7.score(X_test,y_test)
    if (printSuppress != True) :
        print('Gradient Boost Accuracy:',scoreGB)
    
    #Extreme Gradient Boosting has pruning and regularization
    import xgboost as xgb
    model=xgb.XGBClassifier()
    model.fit(X_train, y_train)
    scoreXGB = model.score(X_test,y_test)
    if (printSuppress != True) :
        print('XGBoost:',scoreXGB)
    
    return scoreAB, scoreGB, scoreXGB
    
    
    

def executeSVM(X_train, X_test, y_train, y_test, printSuppress=False): 

    # SVM - Support Vector Machine
    from sklearn import svm
    model8 = svm.SVC(gamma='scale')
    #Run SVM and calculate confidence
    model8.fit(X_train, y_train)
    scoreSVM = model8.score(X_test, y_test)
    if (printSuppress != True) :
        print ('SVM Accuracy:',scoreSVM)

    return scoreSVM
    
    


''' ----------------- Run Everything we know  -------------------

Runs all the Algorithms and returns a dictionary 
Algotype: Accuracy
'''

def executeAllTypes(X_train, X_test, y_train, y_test):


    Algo_Accuracy = {}
    
    confidenceLR, confidenceR, confidenceL = executeRegressionTypes(X_train, X_test, y_train, y_test)
    
    Algo_Accuracy['LiR'] = confidenceLR
    Algo_Accuracy['RR'] =  confidenceR
    Algo_Accuracy['LaR'] = confidenceL
    
    accuracyKNN, accuracyDTTest = executeClassificationTypes(X_train, X_test, y_train, y_test)
    
    Algo_Accuracy['KNN'] = accuracyKNN
    Algo_Accuracy['DT'] =  accuracyDTTest
      
    scoreB, scoreRF, scoreLR, scoreDT, scoreKNN, scoreV_LR_DT_KN, scoreV_BC_RF = executeEnsembleTypes(X_train, X_test, y_train, y_test)
    
    Algo_Accuracy['BC'] = scoreB
    Algo_Accuracy['RF'] =  scoreRF
    Algo_Accuracy['LoR'] = scoreLR
    Algo_Accuracy['V1'] =  scoreV_LR_DT_KN
    Algo_Accuracy['V2'] =  scoreV_BC_RF
    
    scoreAB, scoreGB, scoreXGB = executeBoostingTypes(X_train, X_test, y_train, y_test)
    
    Algo_Accuracy['AB'] = scoreAB
    Algo_Accuracy['GB'] =  scoreGB
    Algo_Accuracy['XGB'] = scoreXGB
    
    scoreSVM = executeSVM(X_train, X_test, y_train, y_test)
    
    Algo_Accuracy['SVM'] = scoreSVM
    
    return Algo_Accuracy


def executeAllClassification(X_train, X_test, y_train, y_test, NTimes):
    
    Algo_Accuracy = {'KNN':[], 'DT':[],'BC':[], 'RF':[], 'LoR':[], 'V1':[], 'V2':[], 'AB':[], 'GB':[], 'XGB':[], 'SVM':[]  }
    
    
    print('Executing Classification Algos for Iterations =', NTimes, '\nPlease Wait ...... \n' )
    
    
    for i in range(NTimes):
    
        print('.',i,end =" ")
        accuracyKNN, accuracyDTTest = executeClassificationTypes(X_train, X_test, y_train, y_test, printSuppress=True)
        
        Algo_Accuracy['KNN'].append(accuracyKNN)
        Algo_Accuracy['DT'].append(accuracyDTTest)
        
        scoreB, scoreRF, scoreLR, scoreDT, scoreKNN, scoreV_LR_DT_KN, scoreV_BC_RF = executeEnsembleTypes(X_train, X_test, y_train, y_test,  printSuppress=True)
        
        Algo_Accuracy['BC'].append(scoreB)
        Algo_Accuracy['RF'].append(scoreRF)
        Algo_Accuracy['LoR'].append(scoreLR)
        Algo_Accuracy['V1'].append(scoreV_LR_DT_KN)
        Algo_Accuracy['V2'].append(scoreV_BC_RF)
        
        scoreAB, scoreGB, scoreXGB = executeBoostingTypes(X_train, X_test, y_train, y_test , printSuppress=True)
        
        Algo_Accuracy['AB'].append(scoreAB)
        Algo_Accuracy['GB'].append(scoreGB)
        Algo_Accuracy['XGB'].append(scoreXGB)
        
        scoreSVM = executeSVM(X_train, X_test, y_train, y_test, printSuppress=True)
        
        Algo_Accuracy['SVM'].append(scoreSVM)
    
    return Algo_Accuracy



''' ----------------- Compare and display results  -------------------'''

def CompareAsBarGraph(Algo_Accuracy):

    import matplotlib.pyplot as plt
    
    plt.bar(range(len(Algo_Accuracy)), list(Algo_Accuracy.values()), align='center')
    plt.xticks(range(len(Algo_Accuracy)), list(Algo_Accuracy.keys()))
    
    plt.show()




def CompareAsMultiLineGraph(Accuracy_Dict):
    
    import matplotlib.pyplot as plt
   
    
    '''
    This dict looks like  
    ( for each Algo there will be N scores, one for each iteration )
    { 'Algo1' : [score1, score2, ... scoreN] ,
      'Algo2' : [score1, score2, ... scoreN] ,
      ...
      'AlgoM' : [score1, score2, ... scoreN]
    }
    '''
    
    colour = ['b','g','r','c','m','y','k','b','g','r','c']
    linestyle = ['solid','dashed']
    lineIndex = 0
    i = 0
    for key, value in Accuracy_Dict.items():
        # plotting the line 1 points 
        if (i > 6):
            lineIndex = 1
        plt.plot(value, color = colour[i], ls=linestyle[lineIndex], label = key)
        i = i+1
        

    plt.xlabel('Iterations')
    # Set the y axis label of the current axis.
    plt.ylabel('Accuracy')
    # Set a title of the current axes.
    plt.title('Compare Classifcation Accuracies')
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    plt.show()


''' -----------------  ACTUALLY DO STUFF -------------------'''

    
df = loadData()

print('All Features in Data Sample \n')
for col in df.columns: 
    print(col) 
    
print(df.head(5))

print('Preprocess and clean up Data \n')
df = preProcessData(df)

# Drop unwanted columns
df = chooseRelevantColumns(df)

print('Features in Use\n')
for col in df.columns: 
    print(col) 

print(df.head(5))

getPrintCorrelations(df)

''' ----------------- Get the Train and Test Split with the Features  -------------------'''


X_train, X_test, y_train, y_test  = UseTwoFeatures(df)
print('----------- Use only 2 most impactful features based on correlation -------------- ')

Algo_Accuracy = executeAllTypes(X_train, X_test, y_train, y_test)
CompareAsBarGraph(Algo_Accuracy)

''' ----------------- Get the Train and Test Split with PCA Reduced Features  -------------------'''
X_train, X_test, y_train, y_test  = UsePCAFeatures(df)
print('----------- Use 4 features reduced to 3 using PCA -------------- ')

PCA_Accuracy = executeAllTypes(X_train, X_test, y_train, y_test)
CompareAsBarGraph(PCA_Accuracy)


''' ----------------- Drop Regression, use only Classification -------------------'''
NTimes = 50
C_Accuracy = executeAllClassification(X_train, X_test, y_train, y_test, NTimes)
CompareAsMultiLineGraph(C_Accuracy)


'''

UnSupervised - try it without the label

We dont have labels in this dataset, so we'll start with unsupervised


# Try 3 Features
Case1features = ['NumRecipients', 'MailTextLen', 'isInternal']

Std_X = df.loc[:, Case1features].values
Std_X = StandardScaler().fit_transform(Std_X)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)
kmeans.fit(Std_X)


# With Mean shift

ms = MeanShift()
ms.fit(Std_X)
labels = ms.labels_
n_clusters_ = len(np.unique(labels))
print("MeanShift Number of estimated clusters:", n_clusters_)

'''
