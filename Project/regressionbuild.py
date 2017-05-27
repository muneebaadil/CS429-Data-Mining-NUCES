
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd 
import sklearn as sk 
import newutils as ut
import os 
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import seaborn
import matplotlib.pyplot as plt 
get_ipython().magic(u'matplotlib notebook')
plt.style.use('ggplot')
reload(ut)


# # Section EDA

# In[2]:

Xdf,Ydf=ut.LoadTrainingSet('./my_training_set/')
Xtraindf,Ytraindf,Xvaldf,Yvaldf=ut.SplitTrainingSet(Xdf,Ydf,.7)
Xtraindf.shape, Ytraindf.shape, Xvaldf.shape, Yvaldf.shape 


# In[3]:

Xtraindf.describe()


# In[4]:

Ytraindf.describe()


# # Section Regression Testing

# In[24]:

class Regressor(object):
    def __init__(self):
        pass
    
    def Train(self,Xtraindf,Ytraindf,Xvaldf,Yvaldf):
        
        #Splitting into two sets w.r.t extreme and normal gap values
        Xtrain1,Ytrain1, Xtrain2, Ytrain2 = self.Preprocess(Xtraindf,Ytraindf,'train')
        Xval1, Yval1, Xval2, Yval2 = self.Preprocess(Xvaldf,Yvaldf,'validation')
        
        nestimator=15
        minsamplesleaf=5#int(.0001*Xtrain.shape[0])
        maxfeatures = 'auto'

        for minsamplesleaf in np.arange(5,5+1,5):
            print 'minsampleleaf:', minsamplesleaf,
            self.reg1=RandomForestRegressor(n_estimators=nestimator, min_samples_leaf=minsamplesleaf,
                                     max_features=maxfeatures)
            self.reg2=RandomForestRegressor(n_estimators=nestimator, min_samples_leaf=minsamplesleaf,
                                     max_features=maxfeatures)
            
            self.reg1.fit(Xtrain1,Ytrain1)
            self.reg2.fit(Xtrain2,Ytrain2)
            
            ypreds1 = self.reg1.predict(Xval1)
            ypreds2 = self.reg2.predict(Xval2)

            #print Yval1.shape, Yval2.shape, ypreds1.shape, ypreds2.shape 
            Yval=np.concatenate((Yval1,Yval2))
            ypreds=np.concatenate((ypreds1,ypreds2))
            losses = np.abs(Yval-ypreds)
            loss = np.mean(losses)

            print 'L1 loss = {}'.format(loss)
        return losses

    def Preprocess(self,Xdf,Ydf,time='train'):
        #1. Features' Construction/Extraction 
        Xdf['Weekday'] = pd.to_datetime(Xdf.Date).dt.weekday
        featstodrop = ['RegionID','Date','Weather','Temperature','PM2.5']    
        Xdf = Xdf.drop(featstodrop,axis=1)
        self.feats = list(Xdf.columns)
        print 'features using = {}'.format(self.feats)
        
        #2. Preprocessing selected features
        if time=='train':
            self.gapmean = Ydf.mean()
            self.gapstd = Ydf.std()
            self.scale = 0.
            X1df,Y1df = Xdf[Ydf <= (self.gapmean+self.scale*self.gapstd)], Ydf[Ydf <= (self.gapmean+self.scale*self.gapstd)]
            X2df,Y2df = Xdf[Ydf > (self.gapmean+self.scale*self.gapstd)], Ydf[Ydf > (self.gapmean+self.scale*self.gapstd)]
            return X1df.values,Y1df.values,X2df.values,Y2df.values
            
        elif time=='validation':
            X1df,Y1df = Xdf[Ydf <= (self.gapmean+self.scale*self.gapstd)], Ydf[Ydf <= (self.gapmean+self.scale*self.gapstd)]
            X2df,Y2df = Xdf[Ydf > (self.gapmean+self.scale*self.gapstd)], Ydf[Ydf > (self.gapmean+self.scale*self.gapstd)]
            return X1df.values,Y1df.values,X2df.values,Y2df.values
        
        elif time=='test':
            return Xdf
        
    def Predict(self,Xdf):
        self.Preprocess(Xdf,None,'test')
        return 


# In[25]:

reg = Regressor()
losses=reg.Train(Xtraindf,Ytraindf,Xvaldf,Yvaldf)


# In[26]:

Xtest=ut.LoadTestSet('./my_test_set/')


# In[27]:

reg.Predict(Xtest)


# In[8]:

fig,ax=plt.subplots()
ax.boxplot(losses)


# In[9]:

Xsuspect=Xval[losses >= (losses.mean()+.5*losses.std())]
Xinnocent=Xval[losses < (losses.mean()+.5*losses.std())]


# In[71]:

pd.DataFrame(Xsuspect,columns=reg.feats).describe()


# In[72]:

pd.DataFrame(Xinnocent,columns=reg.feats).describe()


# In[73]:

def PlotLosses(Xbad, Xgood, feats):
    for i,feat in enumerate(xrange(Xbad.shape[1])):
        plt.figure()
        seaborn.kdeplot(Xbad[:,i],)
        seaborn.kdeplot(Xgood[:,i],)
        plt.title(feats[feat])

ans=PlotLosses(Xsuspect,Xinnocent,reg.feats)


# In[92]:

bins = 2
renge = Xtraindf['F4'].max()/float(bins)
np.unique((Xtraindf['F4']/renge).astype(int))
#np.unique(Xtraindf['F4'])


# In[134]:

Xtraindf.Weather


# In[135]:

Ytraindf.describe()


# In[137]:

Ytraindf[Ytraindf>=(Ytraindf.mean()+Ytraindf.std())].shape[0] / float(Ytraindf.shape[0])

