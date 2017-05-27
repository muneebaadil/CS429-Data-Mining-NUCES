import pdb
from sklearn import linear_model
import numpy as np
import pandas as pd 
import os 
from collections import defaultdict

totimeslot = lambda x: ((int(x[-5:-3]) + (int(x[-8:-6])*60))//10)+1
getregid = lambda x: x.split(',')[2]
getdate = lambda x: x.split(',')[-1]

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

def ConstructTestSet(setname):
    ofnames = [setname+'order_data/'+x for x in sorted(os.listdir('./test_set/order_data/')) if x[0]!='.']
    wfnames = [setname+'weather_data/'+x for x in sorted(os.listdir('./test_set/weather_data/')) if x[0]!='.']
    poifname = setname + 'poi_data/poi_data'
    clustermapfname = setname + 'cluster_map/cluster_map'
    
    print 'files done =',
    i=1
    for ofname, wfname in zip(ofnames, wfnames):
        ConstructTestDay(ofname, wfname, clustermapfname, poifname)
        print str(i)+',',
        i+=1
            
    pass

def ConstructTestDay(ofname, wfname, clusterfname, poifname):
    orderdf=pd.read_csv(ofname,sep=' ',header=None)
    clustermapdf = pd.read_csv(clusterfname, sep='\t', 
                                   names=['RegionHash', 'RegionID'])
    weatherdf = pd.read_csv(wfname, sep='\t', header=None, na_filter=False, 
                         names=['Time', 'Weather', 'Temperature', 'PM2.5'])
    poidf = ConstructPOITable(poifname).reset_index()
    poidf = pd.merge(poidf,clustermapdf,on='RegionHash',how='left')

    orderdf['RegionHash'] = orderdf[0].apply(getregid)
    orderdf['Date'] = orderdf[0].apply(getdate)
    orderdf['Timeslot'] = orderdf[1].apply(totimeslot)
    orderdf.drop([0,1],axis=1,inplace=True)

    orderdf = pd.merge(orderdf,clustermapdf,on='RegionHash',how='left').drop('RegionHash',axis=1)
    weatherdf['Timeslot'] = weatherdf['Time'].apply(totimeslot)
    orderdf=pd.merge(orderdf,weatherdf[['Timeslot','Weather','Temperature','PM2.5']],on='Timeslot',how='left')
    
    temp1=orderdf.groupby(['RegionID','Timeslot','Date'])['Weather','Temperature','PM2.5'].mean().reset_index()
    temp2=orderdf.groupby(['RegionID','Timeslot','Date'])['Weather'].agg({'Demand':lambda x:int(x.shape[0])}).reset_index()
    orderdf=pd.merge(temp1,temp2,on=['RegionID','Timeslot','Date'],how='inner')
    
    orderdf = pd.merge(orderdf,poidf,on='RegionID',how='left').drop('RegionHash',axis=1)
    orderdf.to_csv(np.unique(orderdf.Date.values)[0]+'.csv',index=False)
    return orderdf


def LoadTestSet(dirname):
    fnames = [dirname+x for x in sorted(os.listdir(dirname))]
    Xs,Ys = [],[]
    for fname in fnames: 
        csvfile=pd.read_csv(fname)
        Xs.append(csvfile)
    X = pd.concat(Xs,ignore_index=True)
    X = X[[u'Date', u'RegionID', u'Timeslot', u'Demand', u'Weather',
       u'Temperature', u'PM2.5', u'F1', u'F24', u'F25', u'F20', u'F22', u'F23',
       u'F4', u'F8', u'F5', u'F14', u'F7', u'F15', u'F16', u'F17', u'F11',
       u'F13', u'F19', u'F6', u'F3', u'F2', u'F21', u'F12', u'F18', u'F9',
       u'F10']]
    return X
    
def SaveInKaggleFormat(df, fin, fout):
    clusterfname = './test_set/cluster_map/cluster_map'
    clustermapdf = pd.read_csv(clusterfname, sep='\t', 
                                   names=['RegionHash', 'RegionID'])
    df = pd.merge(df,clustermapdf,on='RegionID',how='left')
    
    myout = df[['Date','RegionHash','Timeslot','gap']]
    myout['id'] = myout.RegionHash.astype(str)+'_'+myout.Date.astype(str)+'_'+(myout['Timeslot']-1).astype(str)
    myout.drop(['Date','RegionHash','Timeslot'],axis=1,inplace=True)
    
    kgout = pd.read_csv(fin)
    kgout['gap']=0.
    
    myout = pd.merge(kgout, myout, on='id', how='left')
    myout = myout.drop('gap_x',axis=1).rename(columns={'gap_y': 'gap'}).fillna(0)
    
    myout.to_csv(fout, index=False)
    return myout