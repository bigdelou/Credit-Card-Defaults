# -*- coding: utf-8 -*-
"""
Created on Mon Jun 3 11:54:03 2019

@author: mbigdelou
"""

#Predicting the default of credit card clients

import os
os.chdir(r'C:\Users\mbigdelou\Desktop\Final Project')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.interactive(False)
import seaborn as sns
import statsmodels.api as sm

#Research Goal: To predict the default of credit card clients based on demografic and historical data
#Reseach Question: Is it possible to predict the default of credit card clients based on the available data? If so how and based on which factors and models?
#Research Hupothesis: 
#1) Demografic Data (Age, Gender, Education, Marrital Status) can predict the default of credit card clients
#2) Historical Data (Amount of the given credit, History of past payment, Amount of bill statement, Amount of previous payment, and the portion of bill statement covered and paid) can predict the default of credit card clients.
#2-1)Amount of the given credit
#2-2)History of past payment (1-6)
#2-3)Amount of bill statement (1-6)
#2-4)Amount of previous payment (1-6)
#2-5)Portion of bill statement covered and paid (1-6)
#3) One of the classifiers (Models) outperform other models based on the accuracy of predictions

#==============Import Dataset
df=pd.read_csv('credit_card_default.csv')
df.head()

df.shape
df.info()
df.dtypes
df.apply(lambda x: sum(x.isnull()))
df.apply(lambda x: len(x.unique()))


list(df)

#checking if any value is missing
df.isnull().any()

# Create new variables based on the available data 
#1 Portion of Bill/Statement Covered/Paid
# Create a new series of variables called df.PORTION_CVRD (Portion of bill/statement paid) for all the 6-month lags
# where the value is 1 if df.BILL_AMT1 is LE 0 and df.PAY_AMT/df.BILL_AMT if not
df['PORTION_CVRD1'] = np.where(df['BILL_AMT1']<=0, 1, df.PAY_AMT1 / df.BILL_AMT1)
df['PORTION_CVRD2'] = np.where(df['BILL_AMT2']<=0, 1, df.PAY_AMT2 / df.BILL_AMT2)
df['PORTION_CVRD3'] = np.where(df['BILL_AMT3']<=0, 1, df.PAY_AMT3 / df.BILL_AMT3)
df['PORTION_CVRD4'] = np.where(df['BILL_AMT4']<=0, 1, df.PAY_AMT4 / df.BILL_AMT4)
df['PORTION_CVRD5'] = np.where(df['BILL_AMT5']<=0, 1, df.PAY_AMT5 / df.BILL_AMT5)
df['PORTION_CVRD6'] = np.where(df['BILL_AMT6']<=0, 1, df.PAY_AMT6 / df.BILL_AMT6)

#2 Portion of Available Credit (df.LIMIT_BAL) outstanding
# Create a new series of variables called df.PORTION_OUTSDNG (Portion of Available Credit (df.LIMIT_BAL) outstanding) for all the 6-month lags
# (df.BILL_AMT - dfPAY_AMT)/df.BILL_AMT if not
df['PORTION_OUTSDNG1'] = np.where(df['BILL_AMT1']<=0, 0, (df['BILL_AMT1'] - df['PAY_AMT1'])/df['LIMIT_BAL'])
df['PORTION_OUTSDNG2'] = np.where(df['BILL_AMT2']<=0, 0, (df['BILL_AMT2'] - df['PAY_AMT2'])/df['LIMIT_BAL'])
df['PORTION_OUTSDNG3'] = np.where(df['BILL_AMT3']<=0, 0, (df['BILL_AMT3'] - df['PAY_AMT3'])/df['LIMIT_BAL'])
df['PORTION_OUTSDNG4'] = np.where(df['BILL_AMT4']<=0, 0, (df['BILL_AMT4'] - df['PAY_AMT4'])/df['LIMIT_BAL'])
df['PORTION_OUTSDNG5'] = np.where(df['BILL_AMT5']<=0, 0, (df['BILL_AMT5'] - df['PAY_AMT5'])/df['LIMIT_BAL'])
df['PORTION_OUTSDNG6'] = np.where(df['BILL_AMT6']<=0, 0, (df['BILL_AMT6'] - df['PAY_AMT6'])/df['LIMIT_BAL'])

# Dropping Unwanted Variables
df = df.drop(['ID'], axis = 1)

df.shape

#========================exploration of variables by univarate measures
describ = df.describe()

from scipy import stats
##LIMIT_BAL
sns.distplot(df.LIMIT_BAL);
plt.show()

sns.violinplot(y='LIMIT_BAL',hue='DEFAULT_PYMNT_NXT_MTH',data=df)
plt.show()

##SEX
df['SEX'].value_counts()

sns.set(rc={'figure.figsize':(10,8)})
sns.countplot(data=df,x='SEX',hue='DEFAULT_PYMNT_NXT_MTH')
plt.show()

df['SEX'] = np.where(df['SEX']=='male', 1,0)

##EDUCATION
df['EDUCATION'].value_counts()

sns.boxplot(x="EDUCATION", y="LIMIT_BAL", hue="DEFAULT_PYMNT_NXT_MTH", data=df, palette="Set3")
plt.show()

sns.set(rc={'figure.figsize':(10,8)})
sns.countplot(data=df,x='EDUCATION',hue='DEFAULT_PYMNT_NXT_MTH')
plt.show()

#Replace Values Encoding
replace_map = {'EDUCATION': {'graduate_school': 1, 'university': 2, 'high_school': 3, 'other': 4}}
df.replace(replace_map, inplace=True)


##MARRIAGE
df['MARRIAGE'].value_counts()

sns.set(rc={'figure.figsize':(10,8)})
sns.countplot(data=df,x='MARRIAGE',hue='DEFAULT_PYMNT_NXT_MTH')
plt.show()

#Replace Values Encoding
replace_map = {'MARRIAGE': {'married': 1, 'single': 2, 'other': 3}}
df.replace(replace_map, inplace=True)

##PAY Features
df['PAY_0'].value_counts()
#for Pay_features if it is <= 0 then it means it was not delayed
Pay_features = ['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
for p in Pay_features:
    df2.loc[df2[p]<=0, p] =0

describ = df.describe()


##PORTION_CVRD
#sns.violinplot(y='PORTION_CVRD1',data=df2, orient='v', inner='quartile')
plt.show()

##PORTION_OUTSDNG
sns.violinplot(y='PORTION_OUTSDNG1',data=df)


##Numerical variables 'LIMIT_BAL','BILL_AMT', 'PAY_AMT' (dollar values)
#LIMIT_BAL and BILL_AMT
with sns.plotting_context("notebook",font_scale=.8):
    g = sns.pairplot(df[['LIMIT_BAL', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']], 
                 palette='tab20',size=1.5)
plt.show()

#PAY_AMT
with sns.plotting_context("notebook",font_scale=0.7):
    g = sns.pairplot(df[['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']], 
                 palette='tab20',size=1.5)
plt.show()


sns.distplot(df.BILL_AMT1)
plt.show()

##Target Variable
df['DEFAULT_PYMNT_NXT_MTH'].value_counts()
df['DEFAULT_PYMNT_NXT_MTH'].value_counts()/df['DEFAULT_PYMNT_NXT_MTH'].count()

#========================Standardazation
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

#StandardScaler
#StandardScaler = StandardScaler()
RobustScaler = RobustScaler()

column_names_to_normalize = ['LIMIT_BAL', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
colnorm = df[column_names_to_normalize].values
colnorm_scaled = RobustScaler.fit_transform(colnorm)
df_temp = pd.DataFrame(colnorm_scaled, columns=column_names_to_normalize, index = df.index)
df[column_names_to_normalize] = df_temp
del colnorm
del colnorm_scaled
del df_temp

#to save the scaler:
from sklearn.externals import joblib
scaler_filename = "RobustScaler_default.save"
joblib.dump(RobustScaler, scaler_filename) 

##LIMIT_BAL
sns.distplot(df.LIMIT_BAL);
plt.show()

#========================exploration of variables by bivariate measures
plt.scatter(df['LIMIT_BAL'],df['PORTION_OUTSDNG1'], c='r',marker='*')
plt.show()

#========================
#==== Correlation Matrix
correlation = df.corr()
plt.figure(figsize=(14, 12))
heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")
plt.show()


#=============Testing for MultiCollinearity
#Variance Inflation Factors:
#Code for VIF Calculation
#a function to calculate the VIF values

def VIF_cal(mcl):
    import statsmodels.formula.api as smf
    x_vars = mcl
    xvar_names = x_vars.columns
    for i in range(0,len(xvar_names)):
        y=x_vars[xvar_names[i]]
        x=x_vars[xvar_names.drop(xvar_names[i])]
        rsq=smf.ols(formula="y~x", data=x_vars).fit().rsquared
        vif=round(1/(1-rsq),3)        
        print(xvar_names[i], "VIF = ", vif)

#Calculating VIF values using the VIF_cal function and drop varibles with vif>4
mcl = df.copy()
VIF_cal(mcl)
mcl = df.drop(['BILL_AMT2', 'BILL_AMT5','BILL_AMT3','BILL_AMT4','PORTION_OUTSDNG2','PORTION_OUTSDNG5','PORTION_OUTSDNG3','BILL_AMT6','PAY_5','PORTION_OUTSDNG4'], axis = 1)
VIF_cal(mcl)
mcl.shape

mcl.notnull().sum()

#==== Correlation Matrix after MultiCollinearity Test
correlation2 = mcl.corr()
plt.figure(figsize=(14, 12))
heatmap = sns.heatmap(correlation2, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")
plt.show()


#==========
#==== extracting independent and target variables
X = mcl.drop(['DEFAULT_PYMNT_NXT_MTH'], axis = 1)
y = df['DEFAULT_PYMNT_NXT_MTH']

#=========
#====Feature Selection
#Recursive Feature Elimination (RFE)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Feature extraction
logreg = LogisticRegression()
rfe = RFE(logreg, 15)
fit = rfe.fit(X, y)
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))

list(X)

X = X.iloc[:,[0,1,2,3,5,6,7,8,10,11,12,14,15,23,24]]
y = df['DEFAULT_PYMNT_NXT_MTH']

list(X)

X.head()

#==== Correlation Matrix for Selected Features
correlation3 = X.corr()
plt.figure(figsize=(14, 12))
heatmap = sns.heatmap(correlation3, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")
plt.show()


#=========
#==== Spliting dataset into traing  and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25,random_state=0) 


# Creating a confusion matrix.
def CMatrix(CM, labels=['pay','default']):
    df = pd.DataFrame(data=CM, index=labels, columns=labels)
    df.index.name='ACTUAL'
    df.columns.name='PREDICTION'
    df.loc['TOTAL'] = df.sum()
    df['Total'] = df.sum(axis=1)
    return df


# Preparing a DataFrame for evaluation metrics
metrics = pd.DataFrame(index=['accuracy', 'precision', 'recall'],
                        columns=['NULL','LogisticReg', 'KNN', 'RandomForest','AdaBoost', 'SVM', 'NaiveBayes'])

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, precision_recall_curve

#Some Insights/Guidelines for Evaluations of Models
#• Accuracy: the proportion of the total number of predictions that are correct 
#• Precision: the proportion of positive predictions that are actually correct 
#• Recall: the proportion of positive observed values correctly predicted as such 

#In this application: 
#• Accuracy: Overall how often the model predicts correctly defaulters and non-defaulters 
#• Precision: When the model predicts default: how often is correct? 
#• Recall: The proportion of actual defalters that the model will correctly predict as such 

#Which metric should I use? It dependes on which mistake (error) we would like to avoid?
#• False Positive: A person who will pay, but predicted as defaulter 
#• False Negative: A person who default, but predicted as payer 
#False negatives are worse => look for a better recall 

#The Null model: always predict the most common category 

# The Null Model (based on the most common category)
y_pred_test = np.repeat(y_train.value_counts().idxmax(), y_test.size)
metrics.loc['accuracy','NULL'] = accuracy_score(y_pred=y_pred_test, y_true=y_test)
metrics.loc['precision','NULL'] = precision_score(y_pred=y_pred_test, y_true=y_test)
metrics.loc['recall','NULL'] = recall_score(y_pred=y_pred_test, y_true=y_test)


CM = confusion_matrix(y_pred=y_pred_test, y_true=y_test)
CMatrix(CM)

#==== Import Models
#LogisticRegression
#KNeighbors (KNN)
#RandomForest (RandomForestClassifier)
#AdaBoost (AdaBoostClassifier)
#SVM (SVC)
#Gaussian Naive Bayes (GaussianNB)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors  import KNeighborsClassifier 
from sklearn.ensemble  import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm  import SVC #Support-vector_machine
from sklearn.naive_bayes import GaussianNB

logreg = LogisticRegression(n_jobs=-1, random_state=0) 
knn = KNeighborsClassifier(n_neighbors=7, weights='distance') 
rfc = RandomForestClassifier(n_estimators=50, max_depth=20, random_state=0) 
adbc = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=0) 
svmc = SVC(kernel='poly', degree=2, gamma='scale') 
nbc = GaussianNB()

#==== Train Classifier
logreg.fit(X_train,y_train)
knn.fit(X_train,y_train)
rfc.fit(X_train,y_train)
adbc.fit(X_train,y_train)
svmc.fit(X_train,y_train)
nbc.fit(X_train,y_train)

#==== Predict on the test set
y_pred_logreg = logreg.predict(X_test)
y_pred_knn = knn.predict(X_test)
y_pred_rfc = rfc.predict(X_test)
y_pred_adbc = adbc.predict(X_test)
y_pred_svmc = svmc.predict(X_test)
y_pred_nbc = nbc.predict(X_test)

#=================Coefficients
logreg.coef_ 

import statsmodels.api as sm
logit_model = sm.Logit(y_train, sm.add_constant(X_train)).fit()
print (logit_model.summary())

#=================Evaluation
#==== Performance Measures
logreg.score(X_test,y_test)
knn.score(X_test,y_test)
rfc.score(X_test,y_test)
adbc.score(X_test,y_test)
svmc.score(X_test,y_test)
nbc.score(X_test,y_test)

#Overfit or not?!
#logreg.score(X_train,y_train)
#knn.score(X_train,y_train)
#rfc.score(X_train,y_train)
#adbc.score(X_train,y_train)
#svmc.score(X_train,y_train)
#nbc.score(X_train,y_train)

#accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, precision_recall_curve
accuracy_score(y_test,y_pred_logreg)
accuracy_score(y_test,y_pred_knn)
accuracy_score(y_test,y_pred_rfc)
accuracy_score(y_test,y_pred_adbc)
accuracy_score(y_test,y_pred_svmc)
accuracy_score(y_test,y_pred_nbc)

#precision_score
precision_score(y_test,y_pred_logreg)
precision_score(y_test,y_pred_knn)
precision_score(y_test,y_pred_rfc)
precision_score(y_test,y_pred_adbc)
precision_score(y_test,y_pred_svmc)
precision_score(y_test,y_pred_nbc)

#recall_score,
recall_score(y_test,y_pred_logreg)
recall_score(y_test,y_pred_knn)
recall_score(y_test,y_pred_rfc)
recall_score(y_test,y_pred_adbc)
recall_score(y_test,y_pred_svmc)
recall_score(y_test,y_pred_nbc)

#confusion_matrix
confusion_matrix(y_test,y_pred_logreg)
confusion_matrix(y_test,y_pred_knn)
confusion_matrix(y_test,y_pred_rfc)
confusion_matrix(y_test,y_pred_adbc)
confusion_matrix(y_test,y_pred_svmc)
confusion_matrix(y_test,y_pred_nbc)

#classification_report
cr_logreg = classification_report(y_test,y_pred_logreg)
cr_knn = classification_report(y_test,y_pred_knn)
cr_rfc = classification_report(y_test,y_pred_rfc)
cr_adbc = classification_report(y_test,y_pred_adbc)
cr_svmc = classification_report(y_test,y_pred_svmc)
cr_nbc = classification_report(y_test,y_pred_nbc)

#Add to metrics
metrics.loc['accuracy','LogisticReg'] = accuracy_score(y_true=y_test, y_pred=y_pred_logreg)
metrics.loc['accuracy','KNN'] = accuracy_score(y_true=y_test, y_pred=y_pred_knn)
metrics.loc['accuracy','RandomForest'] = accuracy_score(y_true=y_test, y_pred=y_pred_rfc)
metrics.loc['accuracy','AdaBoost'] = accuracy_score(y_true=y_test, y_pred=y_pred_adbc)
metrics.loc['accuracy','SVM'] = accuracy_score(y_true=y_test, y_pred=y_pred_svmc)
metrics.loc['accuracy','NaiveBayes'] = accuracy_score(y_true=y_test, y_pred=y_pred_nbc)


metrics.loc['precision','LogisticReg'] = precision_score(y_true=y_test, y_pred=y_pred_logreg)
metrics.loc['precision','KNN'] = precision_score(y_true=y_test, y_pred=y_pred_knn)
metrics.loc['precision','RandomForest'] = precision_score(y_true=y_test, y_pred=y_pred_rfc)
metrics.loc['precision','AdaBoost'] = precision_score(y_true=y_test, y_pred=y_pred_adbc)
metrics.loc['precision','SVM'] = precision_score(y_true=y_test, y_pred=y_pred_svmc)
metrics.loc['precision','NaiveBayes'] = precision_score(y_true=y_test, y_pred=y_pred_nbc)


metrics.loc['recall','LogisticReg'] = recall_score(y_true=y_test, y_pred=y_pred_logreg)
metrics.loc['recall','KNN'] = recall_score(y_true=y_test, y_pred=y_pred_knn)
metrics.loc['recall','RandomForest'] = recall_score(y_true=y_test, y_pred=y_pred_rfc)
metrics.loc['recall','AdaBoost'] = recall_score(y_true=y_test, y_pred=y_pred_adbc)
metrics.loc['recall','SVM'] = recall_score(y_true=y_test, y_pred=y_pred_svmc)
metrics.loc['recall','NaiveBayes'] = recall_score(y_true=y_test, y_pred=y_pred_nbc)

metrics = metrics*100

# Comparing the models with a bar graph.
fig, ax = plt.subplots(figsize=(10,7))
metrics.plot(kind='barh', ax=ax)
ax.grid();
plt.show()


#==============Adjusting the classification threshold
#Adjusting the precision and recall values for all the models
precision_logreg, recall_logreg, thresholds_logreg = precision_recall_curve(y_true=y_test, probas_pred=logreg.predict_proba(X_test)[:,1])
precision_knn, recall_knn, thresholds_knn = precision_recall_curve(y_true=y_test, probas_pred=knn.predict_proba(X_test)[:,1])
precision_rfc, recall_rfc, thresholds_rfc = precision_recall_curve(y_true=y_test, probas_pred=rfc.predict_proba(X_test)[:,1])
precision_adbc, recall_adbc, thresholds_adbc = precision_recall_curve(y_true=y_test, probas_pred=adbc.predict_proba(X_test)[:,1])
#precision_svmc, recall_svmc, thresholds_svmc = precision_recall_curve(y_true=y_test, probas_pred=svmc.predict_proba(X_test)[:,1])
precision_nbc, recall_nbc, thresholds_nbc = precision_recall_curve(y_true=y_test, probas_pred=nbc.predict_proba(X_test)[:,1])


# Plotting the new values for all the models
fig, ax = plt.subplots(figsize=(15,10))
ax.plot(precision_logreg, recall_logreg, label='LogReg')
ax.plot(precision_knn, recall_knn, label='KNN')
ax.plot(precision_rfc, recall_rfc, label='RandomForest')
ax.plot(precision_adbc, recall_adbc, label='AdaBoost')
ax.plot(precision_nbc, recall_nbc, label='NaiveBayes')
ax.set_xlabel('Precision')
ax.set_ylabel('Recall')
ax.set_title('Precision-Recall Curve')
ax.hlines(y=0.5, xmin=0, xmax=1, color='y')
ax.legend()
ax.grid()
plt.show()

# Plotting the new values for the logistic regression model and the Naive Bayes Classifier model.
fig, ax = plt.subplots(figsize=(15,10))
ax.plot(precision_nbc, recall_nbc, label='NaiveBayes')
ax.plot(precision_logreg, recall_logreg, label='LogReg')
ax.set_xlabel('Precision')
ax.set_ylabel('Recall')
ax.set_title('Precision-Recall Curve')
ax.hlines(y=0.5, xmin=0, xmax=1, color='r')
ax.legend()
ax.grid()
plt.show()

# Creating a confusion matrix for modified Logistic Regression Classifier.
fig, ax = plt.subplots(figsize=(15,10))
ax.plot(thresholds_logreg, precision_logreg[1:], label='Precision')
ax.plot(thresholds_logreg, recall_logreg[1:], label='Recall')
ax.set_xlabel('Classification Threshold')
ax.set_ylabel('Precision, Recall')
ax.set_title('Logistic Regression Classifier: Precision-Recall')
ax.hlines(y=0.6, xmin=0, xmax=1, color='r')
ax.legend()
ax.grid()
plt.show()

#Definition : precision_recall_curve(y_true, probas_pred, pos_label=None, sample_weight=None)
#Type : Present in sklearn.metrics.ranking module


# Adjusting the threshold to 0.2 (or very conservatively 0.1)
y_pred_proba = logreg.predict_proba(X_test)[:,1]
y_pred_test = (y_pred_proba >= 0.2).astype('int')

# Confusion Matrix.
CM = confusion_matrix(y_true=y_test, y_pred=y_pred_test)
print('Recall: ', str(100*recall_score(y_pred=y_pred_test, y_true=y_test)) + '%')
print('Precision: ', str(100*precision_score(y_pred=y_pred_test, y_true=y_test)) + '%')
CMatrix(CM)




#========================================== K-Folds
#==== K-Folds Cross Validation (6-fold cross validation)
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics

scores_logreg = cross_val_score(logreg.fit(X_train,y_train), X_train, y_train, cv=6)
print ('Cross-validated scores (cv=6) logreg:', scores_logreg)

scores_knn = cross_val_score(knn.fit(X_train,y_train), X_train, y_train, cv=6)
print ('Cross-validated scores (cv=6) knn:', scores_knn)

scores_rfc = cross_val_score(rfc.fit(X_train,y_train), X_train, y_train, cv=6)
print ('Cross-validated scores (cv=6) rfc:', scores_rfc)

scores_adbc = cross_val_score(adbc.fit(X_train,y_train), X_train, y_train, cv=6)
print ('Cross-validated scores (cv=6) adbc:', scores_adbc)

#Cross validation not working on svmr
scores_svmc = cross_val_score(svmc.fit(X_train,y_train), X_train, y_train, cv=6)
print ('Cross-validated scores (cv=6) svmc:', scores_svmc)

scores_nbc = cross_val_score(nbc.fit(X_train,y_train), X_train, y_train, cv=6)
print ('Cross-validated scores (cv=6) nbc:', scores_nbc)


#==== Grid Search hyper-parameter tuning 
from sklearn.model_selection import GridSearchCV
# Linear Regression does not need hyper-parameter tuning
# KNN
model_knn1 = KNeighborsClassifier() 

param_dict_knn = {
        'n_neighbors': [5,6,7,9,11], 
        'weights': ['uniform', 'distance'], 
        'leaf_size' : [10,20,25,30,35,40],
        }

model_knn2 = GridSearchCV(model_knn1,param_dict_knn)
model_knn2.fit(X_train,y_train)
model_knn2.best_params_
model_knn2.best_score_

# Random Forest Regressor
model_rfc1 = RandomForestClassifier() 

param_dict_rfc = {
        'n_estimators': [20,30,40,50,60], 
        'max_depth': [10,20,30,40,50],         
        }

model_rfc2 = GridSearchCV(model_rfc1, param_dict_rfc)
model_rfc2.fit(X_train,y_train)
model_rfc2.best_params_
model_rfc2.best_score_

# AdaBoost Regressor
model_adbc1 = AdaBoostClassifier()

param_dict_adbc = {
        'n_estimators': [30,40,50,60,70],        
        'learning_rate' : [.1,1,3,10],
        }

model_adbc2 = GridSearchCV(model_adbc1, param_dict_adbc)
model_adbc2.fit(X_train,y_train)
model_adbc2.best_params_
model_adbc2.best_score_

# SVC
model_svmc1 = SVC()

param_dict_svmc = {
        'gamma': ['auto'],
        'C' : [0.001,0.01,0.1,1,10],
        'kernel' : ['rbf', 'linear','poly', 'sigmoid'],        
        'degree' : [2,3,4,5]
        }

model_svmc2 = GridSearchCV(model_svmc1, param_dict_svmc, cv=None)
model_svmc2.fit(X_train,y_train)
model_svmc2.best_params_
model_svmc2.best_score_







#==============END=========