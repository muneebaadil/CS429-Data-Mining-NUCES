from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd 
import os
from sklearn.svm import SVR
import utils
from scipy.stats import mode 

class Model3(object): 
    
    def __init__(self):
        pass
    
    def Train(self,X,supply,demand,splitfraction,verbose=True):
        X,supply,demand=self.Preprocess(X,supply,demand)

        Xtrain,supplyTrain,demandTrain,Xval,supplyVal,demandVal=utils.SplitByRegion(X,supply,demand,.7)
        
        bestscore = np.inf 
        bestnestimaters = None 
        for nestimator in np.arange(10,20+1,5):
            for depth in np.arange(10,20+1,5):
                model=RandomForestRegressor(nestimator, max_depth=depth)
                model.fit(Xtrain,demandTrain-supplyTrain)

                ypreds = model.predict(Xval)
                ypreds2 = model.predict(Xtrain)
                
                score = utils.MeanAbsoluteError(demandVal-supplyVal,ypreds)
                #score2 = np.sum(np.abs((demandVal-supplyVal)-ypreds))/(21.*144*66)
                score2 = utils.MeanAbsoluteError(demandTrain-supplyTrain,ypreds2)
                print 'nestimator = {}, depth = {}, MAE on CV = {}, MAE on Training = {}'.format( nestimator, depth, score, score2)
                if score < bestscore: 
                    bestscore = score 
                    bestdepth = depth
                    bestnestimaters = nestimator

        self.model = RandomForestRegressor(n_estimators=bestnestimaters, max_depth=bestdepth)
        self.model.fit(X,demand-supply)
        pass
    
    def FeatNormalize(self,X,col): 
        X = X.astype(float)
        small, big = X[:,col].min(), X[:,col].max()
        X[:,col] = (X[:,col]-small)/(big-small)
        return X
        
    def Preprocess(self,X,supply,demand):
        X['Weekday'] = pd.to_datetime(X.Date).dt.weekday
        X['Daytype'] = ((X.Weekday==6) | (X.Weekday==5)).astype(bool)
        
        allfacilities = ['1','24','25','20','22','23','4','8','5','14','7','15','16','17','11','13',
                   '19','6','3','2','21','12','18','9','10']
        
        #for facility in allfacilities:
        #    X[facility] = X[facility] > 0.
            
        #X = X.drop(allfacilities, axis=1)
        X = X.drop(['StartRegionID','Date', 'Weekday'], axis=1)
        
        self.feats = list(X.columns)
        print 'Features using = {}'.format(self.feats)
        
        X = X.values
        X = self.Transformer(X)
        
        if (supply is None) or (demand is None):
            return X
        
        supply = supply.values
        demand = demand.values
        
        return X,supply,demand
        
    def Predict(self,X): 
        X=self.Preprocess(X,None,None)
        return self.model.predict(X)
        
    def Transformer(self, X):
        enc=OneHotEncoder(sparse=False, categorical_features=[self.feats.index('Weather')])
        X = enc.fit_transform(X)
        return X