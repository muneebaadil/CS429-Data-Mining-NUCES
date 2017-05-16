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

def Model1(X, supply, demand, splitfraction, Xtest, verbose=True): 
    """"""
    Xnp,supplynp,demandnp = X.values,supply.values,demand.values
    X=X.drop(['Temperature','PM2.5', 'Weather'],axis=1)
    Xtrain, supplyTrain, demandTrain, Xval, supplyVal, demandVal=utils.SplitByRegion(Xnp,supplynp,demandnp,splitfraction)
    
    #regID_ohe = utils.one_hot_encode(Xtrain[:,0],n_classes=67)[:,1:]
    
    bestscore = np.inf 
    bestmodel = None
    bestdeg = None
    
    for d in np.arange(1,10+1):
        Xd = ModelTransformer(Xtrain, d)
        model=linear_model.RidgeCV(alphas=np.logspace(1,20,5),)
        model.fit(Xd, demandTrain-supplyTrain)

        Xd_ = ModelTransformer(Xval, d)
        ypreds = model.predict(Xd_)
        ypreds2 = model.predict(Xd)

        score = utils.MeanAbsoluteError(ypreds,demandVal-supplyVal)
        if bestscore > score: 
            bestscore = score 
            bestmodel = model 
            bestdeg = d
        
        if verbose==True:
            print 'degree = {}, score = {}'.format(d, score)
    
    Xfull = np.concatenate((Xtrain, Xval))
    supplyFull = np.concatenate((supplyTrain, supplyVal))
    demandFull = np.concatenate((demandTrain, demandVal))
    
    Xfullnew = ModelTransformer(Xfull, bestdeg)
    
    fullmodel = linear_model.RidgeCV()
    fullmodel.fit(Xfullnew, demandFull-supplyFull)
    
    
def ModelTransformer(X, d):
    """One-hot-encodes region number, while raises timeslot to degree d"""
    a = utils.one_hot_encode(X[:,0].astype(int),n_classes=67)[:,1:]
    #c = one_hot_encode(X[:,2],n_classes=10)[:,1:]
    b = X[:,1][:, np.newaxis]
    b = PolynomialFeatures(degree=d).fit_transform(b)
    out=np.concatenate((a, b), axis=1)
    return out 
