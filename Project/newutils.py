import pdb
from sklearn import linear_model
import numpy as np
import pandas as pd 
import os 
from collections import defaultdict

totimeslot = lambda x: ((int(x[-5:-3]) + (int(x[-8:-6])*60))//10)+1

def ConstructPOITable(fname, subcat=False):
    poi = pd.read_csv(fname, header=None, names=['Mess'])
    poi['RegionHash'] = poi.Mess.apply(lambda x: x.split('\t')[0])
    poi['AllFacilities'] = poi.Mess.apply(lambda x: x.split('\t')[1:])
    poi.set_index('RegionHash',inplace=True)
    poi.drop('Mess',axis=1,inplace=True)
    
    
    for idx, row in poi.iterrows():
        fcounts = defaultdict(int)
        fvscount = row.values[0]
        for f in fvscount: 
            fidsubcat = f.split(':')[0]
            fidcat = fidsubcat.split('#')[0]
            fcount = f.split(':')[1]

            fcounts[fidcat] += int(fcount)

            if subcat==False:
                #print 'fidsubcat = {}, fidcat = {}, counts = {}'.format(fidsubcat, fidcat, fcounts[fidcat])
                poi.set_value(idx, 'F'+fidcat, fcounts[fidcat])
            else: 
                poi.set_value(idx, fidsubcat, fcount)

    poi.fillna(0,inplace=True)
    for col in poi.columns:
        if col not in ['AllFacilities','RegionHash']: 
            poi[col] = poi[col].astype(int)
            
    return poi.drop('AllFacilities',axis=1) 

def ConstructTrainingDay(orderfname, weatherfname, clustermapfname, poifname):
    orderdf = pd.read_csv(orderfname, sep='\t', header=None, names=['OrderID', 'DriverID',
                          'PassengerID', 'StartRegionHash', 'DestRegionHash', 'Price', 'Time'], index_col = 0)
    
    weatherdf = pd.read_csv(weatherfname, sep='\t', header=None, na_filter=False, 
                 names=['Time', 'Weather', 'Temperature', 'PM2.5'])

    poidf = pd.read_csv(poifname, names=[x for x in range(152)], index_col=0)
    clustermapdf = pd.read_csv(clustermapfname, sep='\t', names=['RegionHash', 'RegionID'])

    out = pd.merge(orderdf, clustermapdf, left_on='StartRegionHash',right_on='RegionHash',how='left').drop(['Price',
                                                                                                           'DestRegionHash',
                                                                                                           'Price',
                                                                                                           'PassengerID',
                                                                                                           'StartRegionHash',
                                                                                                           'RegionHash'],
                                                                                                          axis=1)
    out['Timeslot']=out.Time.apply(totimeslot)
    out['Date']=out.Time.apply(lambda x:x.split(' ')[0])
    out.drop('Time',axis=1, inplace=True)
    out = out.groupby(['Date','RegionID','Timeslot']).DriverID.agg({'Demand': lambda x:int(x.shape[0]),
                                                             'Gap': lambda x:np.sum(pd.isnull(x))}).reset_index()

    weatherdf['Timeslot'] = weatherdf.Time.apply(totimeslot)
    weatherdf = weatherdf.groupby('Timeslot')['Weather','Temperature','PM2.5'].mean().reset_index()
    out = pd.merge(out, weatherdf, on='Timeslot',how='right')

    poidf = ConstructPOITable('./training_set/poi_data/poi_data').reset_index()
    poidf = pd.merge(poidf, clustermapdf, on='RegionHash')

    out = pd.merge(out, poidf.drop('RegionHash',axis=1), on='RegionID',how='left')
    
    out.to_csv(np.unique(out.Date.values)[0]+'.csv',index=False)
    return out 

def ConstructTrainingSet(setname):
    ofnames = [setname+'order_data/'+x for x in sorted(os.listdir('./training_set/order_data/')) if x[0]!='.']
    wfnames = [setname+'weather_data/'+x for x in sorted(os.listdir('./training_set/weather_data/')) if x[0]!='.']
    poifname = setname + 'poi_data/poi_data'
    clustermapfname = setname + 'cluster_map/cluster_map'
    
    print 'files done =',
    i=1
    for ofname, wfname in zip(ofnames, wfnames):
        ConstructTrainingDay(ofname, wfname, clustermapfname, poifname)
        print str(i)+',',
        i+=1
            
    pass

def LoadTrainingSet(dirname): 
    fnames = [dirname+x for x in sorted(os.listdir(dirname))]
    Xs,Ys = [],[]
    for fname in fnames: 
        csvfile=pd.read_csv(fname)
        Xs.append(csvfile.drop('Gap',axis=1))
        Ys.append(csvfile['Gap'])
    X = pd.concat(Xs,ignore_index=True)
    Y = pd.concat(Ys,ignore_index=True)
    return X,Y

def SplitTrainingSet(X,Y,fraction):
    idx = np.arange(0,X.shape[0])
    np.random.shuffle(idx)
    
    X = X.ix[idx]
    Y = Y.ix[idx]

    splitpoint = int(fraction*X.shape[0])
    
    Xtrain = X.iloc[:splitpoint]
    Ytrain = Y.iloc[:splitpoint]
    
    Xval = X.iloc[splitpoint:]
    Yval = Y.iloc[splitpoint:]
    
    return Xtrain,Ytrain,Xval,Yval