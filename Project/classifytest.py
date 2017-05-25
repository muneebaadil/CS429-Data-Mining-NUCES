
# coding: utf-8

# In[1]:

import os 
import pandas as pd 
import numpy as np 
import utils
reload(utils)


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


# In[ ]:



