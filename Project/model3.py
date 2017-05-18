from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd 
import os
import utils

class Model3(object): 
    
    def __init__(self):
        pass
    
    def Train(self,X,supply,demand,splitfraction,verbose=True):
        X = X.drop(['Date','Weather','Temperature','PM2.5'], axis=1).values
        supply = supply.values
        demand = demand.values
        Xtrain, supplyTrain, demandTrain, Xval, supplyVal, demandVal=utils.SplitByRegion(X,supply,demand,.7)
        
        modeln = RandomForestRegressor()
        modeln.fit(X,demand-supply)
        self.model = modeln
        pass
    
    def Predict(self,X): 
        X = X.drop(['Date','Weather','Temperature','PM2.5'], axis=1).values
        return self.model.predict(X)
        