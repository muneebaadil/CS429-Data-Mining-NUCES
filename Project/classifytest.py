
# coding: utf-8

# In[16]:

import os 
import pandas as pd 
import numpy as np 
import utils
reload(utils)
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB


# In[2]:

wfnames = ['./training_set/weather_data/'+x for x in sorted(os.listdir('./training_set/weather_data/')) if x[0]!='.' and x[-4:]!='.csv']
ofnames = ['./training_set/order_data/'+x for x in sorted(os.listdir('./training_set/order_data/')) if x[0]!='.' and x[-4:]!='.csv']
clusterfname = './training_set/cluster_map/cluster_map'
poifname = './training_set/poi_data/poi_data'


# In[5]:

totimeslot = lambda x: ((int(x[-5:-3]) + (int(x[-8:-6])*60))//10)+1

def ConstructDataset(ofnames,wfnames,clusterfname,poifname):
    ofnames, wfnames = sorted(ofnames), sorted(wfnames)
    Xs = []
    Ys = []
    
    clustermapdf = pd.read_csv(clusterfname, sep='\t', names=['RegionHash', 'RegionID'])
    poi = utils.ConstructPOITable(poifname).reset_index()
    
    for ofname, wfname in zip(ofnames, wfnames):
        print 'reading order file = {}'.format(ofname)
        
        orderdf = pd.read_csv(ofname,sep='\t',header=None, names=['OrderID', 'DriverID',
                          'PassengerID', 'StartRegionHash', 'DestRegionHash', 'Price', 'Time'])
        orderdf['Timeslot'] = orderdf.Time.apply(totimeslot)
        orderdf.drop(['OrderID','PassengerID','Price','Time'],axis=1,inplace=True)
        
        weatherdf = pd.read_csv(wfname, sep='\t', header=None, na_filter=False, 
                 names=['Time', 'Weather', 'Temperature', 'PM2.5'])
        weatherdf['Timeslot'] = weatherdf.Time.apply(totimeslot)
        weatherdf.drop(['Time'],axis=1,inplace=True)

        out = pd.merge(orderdf, weatherdf,on='Timeslot',how='left')
        
        out = pd.merge(out, poi, left_on='StartRegionHash',right_on='RegionHash',how='left').drop('RegionHash',axis=1)
        out = pd.merge(out, poi, left_on='DestRegionHash',right_on='RegionHash',how='left').drop('RegionHash',axis=1)
        
        out = pd.merge(out, clustermapdf, left_on='StartRegionHash',right_on='RegionHash',how='left').drop('RegionHash',axis=1)
        out = pd.merge(out, clustermapdf, left_on='DestRegionHash',right_on='RegionHash',how='left').drop('RegionHash',axis=1)
        
        out.drop(['StartRegionHash','DestRegionHash'],axis=1,inplace=True)
        
        for col in out.columns: 
            if col not in ['DriverID','Temperature','PM2.5']: 
                out[col].fillna(0,inplace=True)
                out[col] = out[col].astype(np.uint32)
                
        out['IsAnswered']= ~pd.isnull(out.DriverID)
        out.drop('DriverID',axis=1,inplace=True)
        
        date = ofname.split('_')[-1]
        out['Date']=date
        out.to_csv(date+'.csv')
        del out 


# In[6]:

ConstructDataset(ofnames,wfnames,clusterfname,poifname)


# In[9]:

fnames = ['my_training_set/'+x for x in sorted(os.listdir('./my_training_set/'))]


# In[40]:

def Train(df):
    df = df[df.RegionID_y!=0] #Dropping NANs
    df['Weekday'] = pd.to_datetime(df['Date']).dt.weekday
    df.drop(['RegionID_y','Date'],axis=1,inplace=True)
    Ytrain=df['IsAnswered'].values
    df.drop('IsAnswered',axis=1,inplace=True)
    
    print df[pd.isnull(df.Weather) | pd.isnull(df.Temperature) | pd.isnull(df['PM2.5'])]
    print 'features using = {}'.format(df.columns)
    Xtrain=df.values
    del df
    weights=np.ones((Xtrain.shape[0],))
    weights[Ytrain==False]=2
    
    classifier.partial_fit(Xtrain,Ytrain,np.array([True,False],dtype=bool))


# In[41]:

classifier=GaussianNB()
for fname in fnames[:-7]: 
    print 'training for fname = {}'.format(fname)
    Train(pd.read_csv(fname,index_col=0))


# In[28]:

for fname in fnames[-3:]:
    df=pd.read_csv(fname,index_col=0)
    df = df[df.RegionID_y!=0] #Dropping NANs
    df['Weekday'] = pd.to_datetime(df['Date']).dt.weekday
    #df.drop([],axis=1,inplace=True)
    Ytest=df['IsAnswered'].values
    Xtest=df.drop(['IsAnswered','Weather','Temperature','PM2.5','RegionID_x','RegionID_y','Date'],
                  axis=1).values
    print 'features using = {}'.format(df.columns)
    
    ypreds=classifier.predict(Xtest)
    print 'accuracy = {}'.format(np.mean(ypreds==Ytest))
    
    gaptruths = ComputeGap(df)
    df['IsAnswered']=ypreds
    gappreds = ComputeGap(df)
    del df
    
    print 'Gap l1 loss =',np.mean(np.abs(gappreds['Gap']-gaptruths['Gap']))


# In[25]:

1-(7/21.)


# In[7]:

def ComputeGap(df):
    return df.groupby(['Date','RegionID_x','Timeslot']).IsAnswered.agg({'Gap': lambda x:np.sum(x==False)}).reset_index()

fnames = ['my_training_set/'+x for x in sorted(os.listdir('./my_training_set/'))]
ans=ComputeGap(pd.read_csv(fnames[0],index_col=0))


# In[33]:

abc=pd.read_csv('./my_training_set/2016-01-01.csv')


# In[34]:

abc[pd.isnull(abc.Weather)]


# In[35]:

del abc


# In[ ]:



