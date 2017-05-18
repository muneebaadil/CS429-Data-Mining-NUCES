#MODEL 1: A SIMPLE POLYNOMIAL REGRESSOR ON FOLLOWING TWO FEATURES 
#
#REGION ID (ONE HOT ENCODED)
#TIMESLOT

import numpy as np
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
import os
import utils

class Model1(object):
    
    def __init__(self): 
        self.model = None
        self.degree = None 
        
    def Train(self, X, supply, demand, splitfraction, Xtest, verbose=True): 
        """
        Trains the best polynomial regressor (based on cross-validation) on 
        one hot encoded region ID, and timeslot 

        Args: 
        X (nexamples, ncols): Pandas dataframe having ALL THE RAW FEATURES
        supply (nexamples,): Pandas dataframe having supply values of each training example
        demand (nexamples,): Pandas dataframe having deamnd values of each training example
        splitfraction ([0-1]): Percentage of dataset split between training and CV set
        verbose: (True/False): If True, displays progress of classifier's score on CV set. Otherwise not

        Returns: None
        """

        X=X.drop(['Temperature','PM2.5', 'Weather', 'Date'],axis=1)
        Xnp,supplynp,demandnp = X.values,supply.values,demand.values
        Xtrain, supplyTrain, demandTrain, Xval, supplyVal, demandVal=utils.SplitByRegion(Xnp,supplynp,demandnp,splitfraction)

        bestscore = np.inf 
        bestmodel = None
        bestdeg = None
        
        self.valscores = []
        self.trainscores = []

        for alpha in np.linspace(0,50+1,25):
            for d in np.arange(1,10+1):
                Xd = self.Transformer(Xtrain, d)
                model=linear_model.Ridge(alpha=alpha)
                model.fit(Xd, demandTrain-supplyTrain)

                Xd_ = self.Transformer(Xval, d)
                ypreds = model.predict(Xd_)
                ypreds2 = model.predict(Xd)

                score = utils.MeanAbsoluteError(ypreds,demandVal-supplyVal)
                score2 = utils.MeanAbsoluteError(ypreds2,demandTrain-supplyTrain)

                self.valscores.append(score) 
                self.trainscores.append(score2)

                if bestscore > score: 
                    bestscore = score 
                    bestmodel = model 
                    bestdeg = d

                if verbose==True:
                    print 'degree = {}, alpha = {}, CV score = {}, Training score = {}'.format(d, alpha, score, score2)

        Xfull = np.concatenate((Xtrain, Xval))
        supplyFull = np.concatenate((supplyTrain, supplyVal))
        demandFull = np.concatenate((demandTrain, demandVal))

        Xfullnew = self.Transformer(Xfull, bestdeg)

        fullmodel = linear_model.RidgeCV(scoring='neg_mean_absolute_error')
        fullmodel.fit(Xfullnew, demandFull-supplyFull)
        
        self.model = fullmodel
        self.degree = bestdeg

    def Predict(self, X):
        """
        """
        X = X.drop(['Date','Weather','Temperature','PM2.5'], axis=1)
        Xnp = X.values 
        Xnew = self.Transformer(Xnp,self.degree)
        ypreds = self.model.predict(Xnew)
        return ypreds
    
    def Transformer(self, X, d):
        """One-hot-encodes region number, while raises timeslot to degree d"""
        a = utils.one_hot_encode(X[:,0].astype(int),n_classes=67)[:,1:]
        #c = one_hot_encode(X[:,2],n_classes=10)[:,1:]
        b = X[:,1][:, np.newaxis]
        b = PolynomialFeatures(degree=d).fit_transform(b)
        out=np.concatenate((a, b), axis=1)
        return out 

if __name__=='__main__':
    pass 