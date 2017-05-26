
# coding: utf-8

# In[58]:

import numpy as np
import pandas as pd 
import sklearn as sk 
import newutils as ut
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt 
get_ipython().magic(u'matplotlib notebook')
plt.style.use('ggplot')
reload(ut)


# # Section EDA

# In[60]:

Xdf,Ydf=ut.LoadTrainingSet('./my_training_set/')
Xtraindf,Ytraindf,Xvaldf,Yvaldf=ut.SplitTrainingSet(Xdf,Ydf,.7)
Xtraindf.shape, Ytraindf.shape, Xvaldf.shape, Yvaldf.shape 


# In[61]:

Xtraindf.describe()


# In[62]:

Ytraindf.describe()


# # Section Regression Testing

# In[70]:

def Train(Xtrain,Ytrain,Xval,Yval):
    reg=RandomForestRegressor()
    reg.fit(Xtrain,Ytrain)
    ypreds = reg.predict(Xval)
    
    losses = np.abs(ypreds-Yval)
    loss = np.mean(losses)
    
    print 'L1 loss = {}'.format(loss)
    return losses

def Preprocess(X,Y):
    featstodrop = ['RegionID','Date','Weather','Temperature','PM2.5']
    outX = X.drop(featstodrop,axis=1).values
    print 'features using = {}'.format(set(X.columns)-set(featstodrop))
    outY = Y.values
    return outX,outY


# In[71]:

print 'preprocessing..'
Xtrain,Ytrain = Preprocess(Xtraindf,Ytraindf)
Xval,Yval = Preprocess(Xvaldf,Yvaldf)

print 'fitting..'
losses=Train(Xtrain,Ytrain,Xval,Yval)

