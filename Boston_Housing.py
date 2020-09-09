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

def main():
    path='/Users/CJ/Documents/Kaggle/Boston housing prices'
    
    name_train='/train.csv'
    name_test='/test.csv'
    
    train = pd.read_csv(path+name_train, engine='python')
    test = pd.read_csv(path+name_test, engine='python')
    
    
    del train['Id']
    del test['Id']
    
    train = pd.get_dummies(train)
    #test = pd.get_dummies(test)
    
    train= train.dropna()
    y=np.log(train['SalePrice'])
    del train['SalePrice']
    
    [X_train, X_test, y_train , y_test ] = train_test_split(train, y,  train_size=0.6 , random_state=42)
    
    
    #Scaling
    
    
    X_scaled = X_train
    
    scaler = StandardScaler()
    
    scaler.fit(X_scaled)
    
    mean=scaler.mean_
    var=scaler.var_
    
    [a,b]=X_scaled.shape
    
    
    X_scaled1 = X_scaled.drop(X_scaled.columns[range(36,b)], axis=1)
    
    X_scaled1 = preprocessing.scale(X_scaled1)
    
    X_scaled[X_scaled.columns[range(0,36)]] = X_scaled1
    
    
    
    
    
    model = LinearRegression()
    model.fit(X_scaled, y_train)
    
    
    
    X_scaled = X_test
    
    [a,b]=X_scaled.shape
    
    X_scaled1 = X_scaled.drop(X_scaled.columns[range(36,b)], axis=1)
    
    X_scaled1 = preprocessing.scale(X_scaled1)
    
    X_scaled[X_scaled.columns[range(0,36)]] = X_scaled1
    
    

    y_pred = model.predict(X_scaled)
    
    print((mean_squared_error(y_test, y_pred))**(1/2))
    

if __name__ == "__main__":
    main()