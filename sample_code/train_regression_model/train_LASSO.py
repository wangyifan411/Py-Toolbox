# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:21:47 2020

@author: Yifan Wang

LASSO example using the Boston Housing dataset
"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 

font = {'family' : 'normal', 'size'   : 15}
matplotlib.rc('font', **font)

from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import RepeatedKFold,train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston

#%% Plotting functions

def cal_path(alphas, model, X_cv_train, y_cv_train, X_cv_test, y_cv_test, fit_int_flag):
    
    '''
    Calculate both RMSE and number of coefficients path for plotting purpose
    '''
    
    RMSE_path = []
    coef_path = []
    
    for j in range(len(X_cv_train)):
        
        test_scores = np.zeros(len(alphas))
        coefs_i = np.zeros(len(alphas))
        
        print('{} % done'.format(100*(j+1)/len(X_cv_train)))
        
        for i, ai in enumerate(alphas):
            
            estimator = model(alpha = ai,  max_iter = 1e7, tol = 0.001, fit_intercept=fit_int_flag, random_state = 0)
            estimator.fit(X_cv_train[j], y_cv_train[j])
            # Access the errors, error per cluster
            test_scores[i] = np.sqrt(mean_squared_error(y_cv_test[j], estimator.predict(X_cv_test[j]))) #RMSE
            coefs_i[i] = len(np.nonzero(estimator.coef_)[0])
        
        RMSE_path.append(test_scores)
        coef_path.append(coefs_i)
    
    RMSE_path = np.transpose(np.array(RMSE_path))
    coef_path = np.transpose(np.array(coef_path))

    
    return RMSE_path, coef_path

def plot_coef_path(alpha, alphas, coef_path):
    '''
    #plot alphas vs the number of nonzero coefficents along the path
    '''
    fig = plt.figure(figsize=(6, 4))
    
    plt.plot(-np.log10(alphas), coef_path, ':', linewidth= 0.8)
    plt.plot(-np.log10(alphas), np.mean(coef_path, axis = 1), 
             label='Average across the folds', linewidth=2)     
    plt.axvline(-np.log10(alpha), linestyle='--' , color='r', linewidth=3,
                label='Optimal alpha') 
    plt.legend(frameon=False, loc='best')
    plt.xlabel(r'$-log10(\lambda)$')
    plt.ylabel("# Nonzero Coefficients ")    
    plt.tight_layout()
    plt.show() 



def plot_RMSE_path(alpha, alphas, RMSE_path):
        
    '''
    #plot alphas vs RMSE along the path
    '''

    fig = plt.figure(figsize=(6, 4))
    
    plt.plot(-np.log10(alphas), RMSE_path, ':', linewidth= 0.8)
    plt.plot(-np.log10(alphas), np.mean(RMSE_path, axis = 1), 
             label='Average across the folds', linewidth=2)  
    plt.axvline(-np.log10(alpha), linestyle='--' , color='r', linewidth=3,
                label='Optimal alpha') 
    
    plt.legend(frameon=False,loc='best')
    plt.xlabel(r'$-log10(\lambda)$')
    plt.ylabel("RMSE")    
    plt.tight_layout()
    plt.show()   
       

    
def plot_performance(X, y, model): 
    
    '''
    #plot parity plot
    '''
    y_predict_all = model.predict(X)
    #y_predict_all = predict_y(pi_nonzero, intercept, J_nonzero)
    
    plt.figure(figsize=(6,4))
    
    fig, ax = plt.subplots()
    ax.scatter(y, y_predict_all, s=60, facecolors='none', edgecolors='r')
    
    plt.xlabel("Measured")
    plt.ylabel("Predicted")
    
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    
    # now plot both limits against eachother
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.tight_layout()
    plt.show()
    
    '''
    #plot error plot
    '''

    plt.figure(figsize=(6,4))
    
    fig, ax = plt.subplots()
    ax.scatter(y, (y_predict_all - y), s = 20, color ='r')
    
    plt.xlabel("Measured")
    plt.ylabel("Error")
    
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    
    # now plot both limits against eachother
    ax.plot(lims, np.zeros(len(lims)), 'k--', alpha=0.75, zorder=0)
    ax.set_xlim(lims)
    #ax.set_ylim(lims)

    plt.tight_layout()
    plt.show()
    
 
def plot_path(X, y, alpha, alphas, RMSE_path, coef_path, model):
    
    '''
    Overall plot function for lasso/elastic net
    '''
    
    plot_coef_path(alpha, alphas, coef_path)
    plot_RMSE_path(alpha, alphas, RMSE_path)
    
    '''
    #make performance plot
    '''
    plot_performance(X, y, model)
    
def predict_y(x, intercept, J_nonzero):
    
    # x is the column in pi matrix or the pi matrix 
    y = np.dot(x, J_nonzero) + intercept
    # the results should be the same as y = lasso_cv.predict(X)
    return y


#%%  Main part 
# Import the data, starting from here
X_init, y_init = load_boston(return_X_y=True)

n_points = X_init.shape[0]

fit_int_flag = False
if not fit_int_flag:
    X = np.ones((X_init.shape[0], X_init.shape[1]+1)) #the first column of pi matrix is set a 1, to be the intercept
    X[:,1:] = X_init  

# Prepare for the true output values - y
y = np.array(y_init)
        
if not fit_int_flag:
    scaler = StandardScaler().fit(X[:,1:])
    X[:,1:] = scaler.transform(X[:,1:])
else: 
    scaler = StandardScaler().fit(X)
    X= scaler.transform(X)

sv = scaler.scale_ # standard deviation for each x variable
mv = scaler.mean_ # mean for each x variable

#%% Preparation before regression
# Train test split, save 10% of data point to the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# The alpha grid used for plotting path
alphas_grid = np.logspace(0, -3, 20)

# Cross-validation scheme                                  
rkf = RepeatedKFold(n_splits = 10, n_repeats = 3, random_state = 0)

# Explicitly take out the train/test set
X_cv_train, y_cv_train, X_cv_test, y_cv_test = [],[],[],[]
for train_index, test_index in rkf.split(X_train):
    X_cv_train.append(X_train[train_index])
    y_cv_train.append(y_train[train_index])
    X_cv_test.append(X_train[test_index])
    y_cv_test.append(y_train[test_index])
    
#%% LASSO regression
'''   
# LassoCV to obtain the best alpha, the proper training of Lasso
'''
lasso_cv  = LassoCV(cv = rkf,  max_iter = 1e7, tol = 0.001, fit_intercept=fit_int_flag, random_state=0)
lasso_cv.fit(X_train, y_train)

# the optimal alpha from lassocv
lasso_alpha = lasso_cv.alpha_
# Coefficients for each term
lasso_coefs = lasso_cv.coef_
# The original intercepts 
lasso_intercept = lasso_cv.intercept_

# Access the errors 
y_predict_test = lasso_cv.predict(X_test)
y_predict_train = lasso_cv.predict(X_train)

# RMSE
lasso_RMSE_test = np.sqrt(mean_squared_error(y_test, y_predict_test))
lasso_RMSE_train = np.sqrt(mean_squared_error(y_train, y_predict_train))




##Use alpha grid prepare for lassopath
lasso_RMSE_path, lasso_coef_path = cal_path(alphas_grid, Lasso, X_cv_train, y_cv_train, X_cv_test, y_cv_test, fit_int_flag)
##lasso_path to get alphas and coef_path, somehow individual CV does not work
plot_path(X, y, lasso_alpha, alphas_grid, lasso_RMSE_path, lasso_coef_path, lasso_cv)


'''
lasso coefficients needed to be tranformed back to the regular form
'''
lasso_coefs_regular = np.zeros(len(lasso_coefs))
lasso_coefs_regular[1:] = lasso_coefs[1:]/sv
lasso_coefs_regular[0] = lasso_coefs[0] - np.sum(mv/sv*lasso_coefs[1:])

#%% Select the significant coefficients
'''
LASSO Post Processing
'''

# Set the tolerance for signficant interactions 
Tol = 1e-5
# The indices for non-zero coefficients
J_index = np.where(abs(lasso_coefs_regular)>Tol)[0]
# The number of non-zero coefficients  
n_nonzero = len(J_index)
# The values of non-zero coefficients
J_nonzero = lasso_coefs_regular[J_index] 
X_nonzero = X[:, J_index]

# Adjust for the manual intercept fitting
if not fit_int_flag:
    
    intercept = J_nonzero[0]
    n_nonzero = n_nonzero - 1 
    J_nonzero = J_nonzero[1:]
    X_nonzero = X_nonzero[:,1:]


        
        

