# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 17:12:26 2019

@author: Brandon Townsend
"""



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def loadData(filename='creditcard.csv', testSize=0.25, shuff=True, strat=None, dump = None):
    data = pd.read_csv(filename)

    if dump != None:
        for feature in dump:
            if feature in data:
                data = data.drop(feature, axis=1)

    splt = np.split(data,[len(data.columns)-1], axis=1)
    #pass into tts as np arrays
    X_train, X_test, y_train, y_test = train_test_split(splt[0].values, splt[1].values, test_size=testSize, shuffle=shuff, stratify = strat,random_state=42)


    return  X_train, X_test, y_train, y_test
#    
    
    

if __name__ == '__main__':
    
    X_train, X_test, y_train, y_test = loadData(testSize=0.1,shuff=True)
    print(X_train, X_test, y_train, y_test)