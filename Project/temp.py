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
        self.enc = None
        pass
    
    def Train(self,X,Y,splitfraction,verbose=True):
        X,Y=self.Preprocess(X,Y)
        Xtrain,Ytrain,Xval,Yval=utils.SplitByRegion(X,Y,.7)
        
        bestscore = np.inf 
        bestnestimaters = None 
        
        for nestimator in np.arange(15,25+1,5):
            for depth in np.arange(15,30+1,5):
                model=RandomForestRegressor(nestimator, max_depth=depth)
                model.fit(Xtrain,Ytrain)

                ypreds = model.predict(Xval)
                ypreds2 = model.predict(Xtrain)
                
                score = utils.MeanAbsoluteError(Yval,ypreds)
                #score2 = np.sum(np.abs((demandVal-supplyVal)-ypreds))/(21.*144*66)
                score2 = utils.MeanAbsoluteError(Ytrain,ypreds2)
                print 'nestimator = {}, depth = {}, MAE on CV = {}, MAE on Training = {}'.format( nestimator, depth, score, score2)
                if score < bestscore: 
                    bestscore = score 
                    bestdepth = depth
                    bestnestimaters = nestimator

        self.model = RandomForestRegressor(n_estimators=bestnestimaters, max_depth=bestdepth)        
        self.model.fit(X,Y)
        return 
    
    def FeatNormalize(self,X,col): 
        X = X.astype(float)
        small, big = X[:,col].min(), X[:,col].max()
        X[:,col] = (X[:,col]-small)/(big-small)
        return X
        
    def Preprocess(self,X,Y):
        X['Weekday'] = pd.to_datetime(X.Date).dt.weekday
        X['Daytype'] = ((X.Weekday==6) | (X.Weekday==5)).astype(bool)
        
        allfacilities = ['1','24','25','20','22','23','4','8','5','14','7','15','16','17','11','13',
                   '19','6','3','2','21','12','18','9','10']
        
        X = X.drop(['StartRegionID','Date'], axis=1)
        
        self.feats = list(X.columns)
        print 'Features using = {}'.format(self.feats)
        
        X = X.values
        #X = self.Transformer(X)
        
        if (Y is None):
            return X
        
        return X,Y.values
        
    def Predict(self,X): 
        #if (pd.isnull(X.Weather)).sum() > 0: #If any missing value 
        #print 'came here\n'
        #X=self.PredictMissingValues(X)
        X=self.Preprocess(X,None)
        return self.model.predict(X)
    
    def PredictMissingValues(self,Xtest):
        """ASSUMPTION: XTEST IS SUPPOSED TO BE OF SINGLE DAY"""
        timeslotpd = pd.DataFrame({'Timeslot': np.arange(1,144+1,1)})
        Xtestg = Xtest.groupby('Timeslot')['Weather','Temperature','PM2.5'].mean().reset_index()
        out = pd.merge(timeslotpd, Xtestg, on='Timeslot', how='left')
        out.fillna(method='ffill', inplace=True)
        out.fillna(method='bfill', inplace=True)
        out = out.set_index('Timeslot')
        Xtest.Weather = out.ix[Xtest.Timeslot.values].Weather.values
        Xtest.Temperature = out.ix[Xtest.Timeslot.values].Temperature.values
        Xtest['PM2.5'] = out.ix[Xtest.Timeslot.values]['PM2.5'].values
        return Xtest
    
    def Transformer(self, X):
        if self.enc is None: 
            self.enc=OneHotEncoder(sparse=False, categorical_features=[self.feats.index('Weather')],
                                  n_values=X[:,self.feats.index('Weather')].max()+1)
        X = self.enc.fit_transform(X)
        #X = X[:,np.sum(X,axis=0)>0]
        return X