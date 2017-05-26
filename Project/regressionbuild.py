
# coding: utf-8

# In[1]:

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

# In[41]:

Xdf,Ydf=ut.LoadTrainingSet('./my_training_set/')
Xtraindf,Ytraindf,Xvaldf,Yvaldf=ut.SplitTrainingSet(Xdf,Ydf,.7)
Xtraindf.shape, Ytraindf.shape, Xvaldf.shape, Yvaldf.shape 


# In[42]:

Xtraindf.describe()


# In[43]:

Ytraindf.describe()


# # Section Regression Testing

# In[103]:

def Train(Xtrain,Ytrain,Xval,Yval):
    
    nestimator=15
    minsamplesleaf=5#int(.0001*Xtrain.shape[0])
    maxfeatures = 'auto'
    
    for minsamplesleaf in np.arange(1,5+1,1):
        print 'minsampleleaf:', minsamplesleaf,
        reg=RandomForestRegressor(n_estimators=nestimator, min_samples_leaf=minsamplesleaf,
                                 max_features=maxfeatures)
        reg.fit(Xtrain,Ytrain)
        ypreds = reg.predict(Xval)

        losses = np.abs(ypreds-Yval)
        loss = np.mean(losses)

        print 'L1 loss = {}'.format(loss)
    return losses

def Preprocess(Xdf,Ydf):
    featstodrop = ['RegionID','Date']    
    Xdf = Xdf.drop(featstodrop,axis=1)
    X = Xdf.values
    Y = Ydf.values
    
    #Removing 'Y-outliers'
    ylimit = np.mean(Y)+np.std(Y)
    X = X[Y<=ylimit]
    Y = Y[Y<=ylimit]
    
    #One hot encoding weather feature
    enc = OneHotEncoder(n_values=10, sparse=False,
                        categorical_features=[list(Xdf.columns).index('Weather')])
    X = enc.fit_transform(X)
    
    
    print 'features using = {}'.format(set(Xdf.columns)-set(featstodrop))
    
    return X,Y


# In[104]:

print 'preprocessing..'
Xtrain,Ytrain = Preprocess(Xtraindf,Ytraindf)
Xval,Yval = Preprocess(Xvaldf,Yvaldf)

print '\nfitting..'
losses=Train(Xtrain,Ytrain,Xval,Yval)


# In[72]:

Xsuspect=Xval[losses>=np.percentile(losses,75)]


# In[94]:

Ytraindf.describe()


# In[102]:

Ytraindf[Ytraindf>=()].shape[0]/float(Ytraindf.shape[0])


# In[101]:

iqr=np.percentile(Ytraindf,75)-np.percentile(Ytraindf,25)
np.percentile(Ytraindf,50)+(1.5*iqr)


# In[ ]:



