#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 19:42:33 2020

@author: CJ
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

def load_prepp_data(path='/Users/CJ/Documents/Kaggle/Boston housing prices',
              name_train='/train.csv',
              name_test='/test.csv'):
    
    
    train = pd.read_csv(path+name_train, engine='python')
    test = pd.read_csv(path+name_test, engine='python')
    
    
    del train['Id']
    del test['Id']
    train = pd.get_dummies(train)
    #test = pd.get_dummies(test)
    train = train.dropna()
    
    print(train.shape)
    y=np.log(train['SalePrice'])
    del train['SalePrice']
    
    [X_train, X_test, y_train , y_test ] = train_test_split(train, 
                                                            y,  
                                                             train_size=0.6,
                                                            random_state=42)
    return X_train, X_test, y_train , y_test

def model(X=X_train, y=y_train):
    model = LinearRegression()
    return model.fit(X, y)


def predict(X_test=X_test, y_test=y_test): 
    
    y_pred = model.predict(X_test)
    result = (mean_squared_error(y_test, y_pred))**(1/2)
    
    return result
  
    

if __name__ == "__main__":
    X_train, X_test, y_train , y_test = load_prepp_data()
    model(X_train,y_train)
    result = predict(X_test,y_test) 
    print(result)