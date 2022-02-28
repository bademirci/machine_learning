# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 17:25:16 2021

@authors: 
Ahmet Bakkal - 070190139
Batuhan Demirci - 070190155
"""
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"Importing data from HW1DataSet.csv."
df = pd.read_csv("HW1DataSet.csv")

y = np.array(df["y"].values)
x = np.array(df["x"].values.reshape(-1,1))


"(a) Split your dataset into a test set and train set. (20% test set, 80% training set)"

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=5)

def create_polynomial_regression_model(degree):
  "Creates a polynomial regression model for the given degree"
  
  poly_features = PolynomialFeatures(degree=degree,include_bias=False)
  
  # transforms the existing features to higher degree features.
  X_train_poly = poly_features.fit_transform(X_trainv)
  
  # fit the transformed features to Linear Regression
  poly_model = LinearRegression()
  poly_model.fit(X_train_poly, Y_trainv)
  
  # predicting on training data-set
  y_train_predicted = poly_model.predict(X_train_poly)
  
  # predicting on validation data-set
  y_valid_predict = poly_model.predict(poly_features.fit_transform(X_valid))
  
  # evaluating the model on training dataset
  mse_train = mean_squared_error(Y_trainv, y_train_predicted)
  
  # evaluating the model on validation dataset
  mse_valid = mean_squared_error(Y_valid, y_valid_predict)
   
  return (mse_train,mse_valid,degree)

"(b) Try different degrees of polynomial functions and pick the one that has the"
"smallest LOOCV mean squared error, and report LOOCV validation errors of"
"each polynomial functions"

loo = LeaveOneOut()
LeaveOneOutCV=[]
for j in range(1,8):
    LeaveOneOutErrors=[]
    for train_index, validation_index in loo.split(X_train):
        X_trainv, X_valid = X_train[train_index],X_train[validation_index]
        Y_trainv, Y_valid = Y_train[train_index], Y_train[validation_index]
        LeaveOneOutErrors.append(create_polynomial_regression_model(j))
        df = pd.DataFrame (LeaveOneOutErrors,columns=['Training Error','Validation Error','Degree'])
    LeaveOneOutCV.append([df['Validation Error'].mean(),df['Degree'][0]])  
LeaveOneOutCV_df = pd.DataFrame (LeaveOneOutCV,columns=['Validation Error','Degree'])

print(LeaveOneOutCV_df)
print("-----------------------------")



minimum_index = LeaveOneOutCV_df["Validation Error"].idxmin()
print(LeaveOneOutCV_df.loc[minimum_index,"Degree"],"degree polynomial function has the smallest LOOCV mean squared error.")
print("-----------------------------")


"(c) Refit your model on the training set with the selected degree of polynomial"
"and compute the test mean squared error"

poly_features = PolynomialFeatures(degree=3,include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
poly_model = LinearRegression()
poly_model.fit(X_train_poly, Y_train)
X_test_poly = poly_features.fit_transform(X_test)
Y_test_predict=poly_model.predict(X_test_poly)
mse_test = mean_squared_error(Y_test, Y_test_predict)
print("Mean Squared Error:",mse_test)

"(d) Try different degrees of polynomial functions and pick the one that has the"
"smallest 5-fold cross validation mean squared error, and report 5-fold cross"
"validation errors of each polynomial functions."

cv = KFold(n_splits=5, random_state=42, shuffle=True)
CVErrors=[]
for train_index, validation_index in cv.split(X_train):
    X_trainv, X_valid, Y_trainv, Y_valid = X_train[train_index], X_train[validation_index], Y_train[train_index], Y_train[validation_index]
    for j in range(1,8):
        CVErrors.append(create_polynomial_regression_model(j))
        kFold_df = pd.DataFrame (CVErrors,columns=['Training Error','Validation Error','Degree'])


kfoldCV_by_degree = kFold_df.groupby("Degree")
kfoldCV_by_degree = kfoldCV_by_degree.mean()
kfoldCV_by_degree = kfoldCV_by_degree.reset_index()

print("-----------------------------")
print(kfoldCV_by_degree[['Degree', 'Validation Error']])
print("-----------------------------")



minimum_index = kfoldCV_by_degree["Validation Error"].idxmin()
print(kfoldCV_by_degree.loc[minimum_index,"Degree"],"degree polynomial function has the smallest 5-fold cross validation mean squared error.")
print("-----------------------------")


"(e) Refit your model on the training set with the selected degree of polynomial"
"and compute the test mean squared error and test R2 score."

r2_calc = r2_score(Y_test, Y_test_predict)
print("R2 Score:",r2_calc)

"(f) Are the degrees of polynomials chosen with LOOCV and 5-fold cross"
"validation same?"
print("According to the results of both LOOCV and K-Fold Cross Validation, we got 3 as the polynomial degree which has minimum validation mse error.")
print("-----------------------------")

print("Graphs:")
fig=plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(LeaveOneOutCV_df['Degree'].values,LeaveOneOutCV_df['Validation Error'].values,label = 'Leave One Out Error')
ax.set_xlabel('Degree')
ax.set_ylabel('LOOC Errors')
ax.tick_params(axis='x', labelsize=8)
ax.legend(loc='best')


ax.plot(kfoldCV_by_degree['Degree'].values,kfoldCV_by_degree['Validation Error'].values,label = '5 fold CV Error', linewidth=2)
ax.tick_params(axis='x', labelsize=8)
ax.set_ylabel('5-Fold Errors')
ax.legend(loc='best')
plt.show()

