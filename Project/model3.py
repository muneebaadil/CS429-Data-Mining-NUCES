from sklearn.ensemble import RandomForestRegressor
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
        
        X = X.copy()
        supply = supply.copy()
        demand = demand.copy()
        
        X,supply,demand=self.Preprocess(X,supply,demand)

        Xtrain,supplyTrain,demandTrain,Xval,supplyVal,demandVal=utils.SplitByRegion(X,supply,demand,.7)
        
        bestscore = np.inf 
        bestnestimaters = None 
        for nestimator in np.arange(10,20+1,5):
            for depth in np.arange(5,30+1,5):
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
        X = X.drop(['1','24','25','20','22','23','4','8','5','14','7','15','16','17','11','13',
                   '19','6','3','2','21','12','18','9','10'],axis=1)
        X = X.drop(['Temperature', 'Weather', 'PM2.5', 'Date', 'Weekday'], axis=1)
        
        print 'Features using = {}'.format(X.columns)
        
        X = X.values
        if (supply is None) or (demand is None):
            return X
        supply = supply.values
        demand = demand.values
        
        return X,supply,demand
        
    def Predict(self,X): 
        X=self.Preprocess(X,None,None)
        return self.model.predict(X)
        
    def Transformer(self, X):
        """One-hot-encodes region number"""    
        #regionOHE = utils.one_hot_encode(X[:,0].astype(int),n_classes=67)[:,1:]
        weatherOHE = utils.one_hot_encode(X[:,2].astype(int),n_classes=7)
        weatherOHE = weatherOHE[:, weatherOHE.sum(axis=0)!=0]
        #out = np.concatenate((regionOHE, X[:,1][:,np.newaxis], weatherOHE),axis=1)
        out = np.concatenate((X[:,:2],weatherOHE),axis=1)
        
        return out 