#MODEL 2: PIECE-WISE POLYNOMIAL REGRESSOR ON TWO FOLLOWING FEATURES 
#
#1. ONE-HOT-ENCODED REGION ID 
#2. TIMESLOTS

import numpy as np
import pandas as pd 
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import os
import utils

class Model2(object): 
    
    def __init__(self): 
        models = list() 
        timesplits = list() 
        
    def Transformer(self, X, d):
        """One-hot-encodes region number, while raises timeslot to degree d"""
        a = utils.one_hot_encode(X[:,0].astype(int),n_classes=67)[:,1:]
        #c = one_hot_encode(X[:,2],n_classes=10)[:,1:]
        b = X[:,1][:, np.newaxis]
        b = PolynomialFeatures(degree=d).fit_transform(b)
        out=np.concatenate((a, b), axis=1)
        return out 
        
    def SplitSetByTimeslot(self, X, supply, demand, splits):
        X1 = X[X[:,1]<=splits[0]]
        S1 = supply[X[:,1]<=splits[0]]
        D1 = demand[X[:,1]<=splits[0]]
        
        X2 = X[(X[:,1]<=splits[1]) & (X[:,1]>splits[0])]
        S2 = supply[(X[:,1]<=splits[1]) & (X[:,1]>splits[0])]
        D2 = demand[(X[:,1]<=splits[1]) & (X[:,1]>splits[0])]
        
        X3 = X[X[:,1]>splits[1]]
        S3 = supply[X[:,1]>splits[1]]
        D3 = demand[X[:,1]>splits[1]]  
        
        return X1, S1, D1, X2, S2, D2, X3, S3, D3
    
    def Train(self,X,supply,demand,splitfraction,verbose=True):
        
        timesplits = [(x,y) for x in np.arange(10,142+1,10) for y in np.arange(x+10,143+1,10)]
        
        X=X.drop(['Temperature','PM2.5', 'Weather', 'Date'],axis=1)
        Xnp,supplynp,demandnp = X.values,supply.values,demand.values
        Xtrain, supplyTrain, demandTrain, Xval, supplyVal, demandVal=utils.SplitByRegion(Xnp,supplynp,demandnp,splitfraction)
        
        for splits in timesplits: 
            for d in np.arange(1,10+1):
                for alpha in np.linspace(0,50,10): 
                    
                    X1train, S1train, D1train, X2train, S2train, D2train, X3train, S3train, D3train = self.SplitSetByTimeslot(Xtrain, supplyTrain, demandTrain, splits)
                    
                    X1train = self.Transformer(X1train, d)
                    X2train = self.Transformer(X2train, d)
                    X3train = self.Transformer(X3train, d)
                    
                    m1, m2, m3 = linear_model.Ridge(alpha=alpha), linear_model.Ridge(alpha=alpha), linear_model.Ridge(alpha=alpha)
                    m1.fit(X1train, D1train-S1train) 
                    m2.fit(X2train, D2train-S2train) 
                    m3.fit(X3train, D3train-S3train)
                    
                    print 'splits = {}, d = {}, alpha = {}'.format(splits,d,alpha)
                    
        return 
    
    def Predict(self, X, splits=None): 
        if splits is None: 
            pass 
        elif: 
            pass
        pass
    