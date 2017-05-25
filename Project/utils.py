import pdb
from sklearn import linear_model
import numpy as np
import pandas as pd 
import os 
from collections import defaultdict

def SplitByRegion(X, Y, splitfrac):
    Xtrain, Ytrain = [], []
    Xval, Yval = [], []
    
    regions = np.unique(X[:, 0])

    for r in regions: 
        Xr, Yr = X[X[:, 0]==r], Y[X[:, 0]==r]
        
        randidx = np.random.choice(Xr.shape[0], Xr.shape[0], replace=False)
        splitidx = int(splitfrac * Xr.shape[0])
        
        Xtrain.append(Xr[randidx[:splitidx]])
        Ytrain.append(Yr[randidx[:splitidx]])
        
        Xval.append(Xr[randidx[splitidx:]])
        Yval.append(Yr[randidx[splitidx:]])
        
    Xtrain, Ytrain = np.concatenate(Xtrain), np.concatenate(Ytrain)
    Xval, Yval = np.concatenate(Xval), np.concatenate(Yval)
    
    return Xtrain, Ytrain, Xval, Yval

def LoadTrainingSet(basepath):
    fnames = [x for x in sorted(os.listdir(basepath)) if x[0]!='.']
    #X, supply, demand = None, None, None
    Xs, Ys = [], []
    for fname in fnames: 
        table = pd.read_csv(basepath+fname)
        Ys.append(table['Demand']-table['Supply'])
        Xs.append(table.drop(['Supply'], axis=1))
        
    Y = pd.concat(Ys)
    X = pd.concat(Xs)

    return X, Y

totimeslot = lambda x: ((int(x[-5:-3]) + (int(x[-8:-6])*60))//10)+1

def ConstructTrainingSet(setname, verbose=True):
    ofnames = [setname+'order_data/'+x for x in sorted(os.listdir('./training_set/order_data/')) if x[0]!='.']
    wfnames = [setname+'weather_data/'+x for x in sorted(os.listdir('./training_set/weather_data/')) if x[0]!='.']
    poifname = setname + 'poi_data/poi_data'
    clustermapfname = setname + 'cluster_map/cluster_map'
    
    for ofname, wfname in zip(ofnames, wfnames):
        
        date = ofname.split('_')[-1]
        outname = setname+'DesignMatrices/'+date+'.csv'
        ConstructTrainingDay(ofname, wfname, poifname, clustermapfname, date, outname)
        if verbose==True:
            print 'Construction for training day =', date
    pass

def ConstructTrainingDay(orderfname, weatherfname, poifname, clustermapfname, date, outname):    
    #Reading all files for the day given as argument
    orderdf = pd.read_csv(orderfname, sep='\t', header=None, na_filter=False, names=['OrderID', 'DriverID',
                          'PassengerID', 'StartRegionHash', 'DestRegionHash', 'Price', 'Time'], index_col = 0)
    
    weatherdf = pd.read_csv(weatherfname, sep='\t', header=None, na_filter=False, 
                 names=['time', 'weather', 'temperature', 'pm2.5'])
    
    poidf = pd.read_csv(poifname, names=[x for x in range(152)], index_col=0)
    clustermapdf = pd.read_csv(clustermapfname, sep='\t', names=['RegionHash', 'RegionID'], index_col = 0)
    
    #Mapping region hash values to problem-specific region codes 
    orderdf['StartRegionID'] = clustermapdf.ix[orderdf['StartRegionHash'].values].values[:,0]
    
    #Adding timeslot of each order 
    orderdf['Timeslot'] = orderdf['Time'].apply(totimeslot)
    
    #Adding weather condition to each order
    weatherdf['timeslot'] = weatherdf['time'].apply(totimeslot)
    orderdf['Weather'] = (((weatherdf.groupby('timeslot')['weather'].mean()).ix[orderdf['Timeslot'].values]).values).astype(int)
    
    #Adding temperature info, and pollution information. 
    orderdf['Temperature'] = (weatherdf.groupby('timeslot')['temperature'].mean()).ix[orderdf['Timeslot'].values].values 
    orderdf['PM2.5'] = (weatherdf.groupby('timeslot')['pm2.5'].mean()).ix[orderdf['Timeslot'].values].values
    orderdf['Date'] = date
      
    #Constructing a design matrix to store 
    designMatrix = (orderdf.groupby(['StartRegionID', 'Timeslot', 'Weather', 'Temperature', 'PM2.5', 'Date']))['DriverID'].agg({
                    'Demand': 'count', 'Supply': lambda x:np.sum(x!='NULL')})
    
    #Finally, adding POI information 
    poidf = ConstructPOITable(poifname, False)
    poidf['StartRegionID'] = clustermapdf.ix[poidf.index.values].values[:,0]
    
    designMatrix = designMatrix.reset_index()
    designMatrix=designMatrix.merge(poidf, on='StartRegionID')
    
    #Saving the constructed design matrix
    designMatrix.to_csv(outname, sep=',', index=False)
    return designMatrix

getdate = lambda x: x.split(',')[-1]
getregid = lambda x: x.split(',')[2]

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
                poi.set_value(idx, fidcat, fcounts[fidcat])
            else: 
                poi.set_value(idx, fidsubcat, fcount)

    poi.fillna(0,inplace=True)
    for col in poi.columns:
        if col not in ['AllFacilities','RegionHash']: 
            poi[col] = poi[col].astype(int)
            
    return poi.drop('AllFacilities',axis=1) 

def ConstructWeatherVectors(foldername, outfolder):
    fnames = [x for x in sorted(os.listdir(foldername)) if x[0]!='.']
    for fname in fnames: 
        print 'fname =', fname
        wdf = pd.read_csv(foldername+fname, sep='\t', 
                          names=['Timestamp', 'Weather','Temperature', 'PM2.5'], header=None)
        wdf['Timeslot']=wdf.Timestamp.apply(totimeslot)
        
        wdf.groupby('Timeslot').Weather.mean().astype(int).to_csv(outfolder+'wvector_'+fname.split('_')[2]+'.csv')
    return

def LoadWeatherMatrix(foldername, fillmethod='backfill'):
    fnames = [x for x in sorted(os.listdir(foldername)) if x[:7]=='wvector' and x[-4:]=='.csv']
    dfs = []
    for fname in fnames:
        #print 'fname = {}'.format(fname)
        df= pd.read_csv(foldername+fname, header=None, names=['Timeslot', fname.split('_')[1]], index_col='Timeslot')
        dfs.append(df)
        
    df = pd.concat(dfs, axis=1)
    if fillmethod is not None:
        df=df.fillna(method=fillmethod, axis=1)
    return df
    pass

def ConstructTestSet(setname, verbose=True):
    ofnames = [setname+'order_data/'+x for x in sorted(os.listdir(setname+'order_data/')) if x[0]!='.']
    wfnames = [setname+'weather_data/'+x for x in sorted(os.listdir(setname+'weather_data/')) if x[0]!='.']
    poifname = setname + 'poi_data/poi_data'
    clustermapfname = setname + 'cluster_map/cluster_map'
    
    for ofname, wfname in zip(ofnames, wfnames):
        
        date = ofname.split('_')[-1]
        outname = setname+'DesignMatrices/'+date+'.csv'
        ConstructTestDay(ofname, wfname, poifname, clustermapfname, date, outname)
        if verbose==True:
            print 'Construction for training day =', date
    pass

def ConstructTestDay(orderfname, weatherfname, poifname, clustermapfname, date, outname):
    
    #Reading all files for the filename given as argument 
    orderdf = pd.read_csv(orderfname, sep=' ', header=None)
    weatherdf = pd.read_csv(weatherfname, sep='\t', header=None, na_filter=False, 
                     names=['Time', 'Weather', 'Temperature', 'PM2.5'])
    clustermapdf = pd.read_csv(clustermapfname, sep='\t', 
                               names=['RegionHash', 'RegionID'], index_col = 0)

    orderdf['RegionHash'] = orderdf[0].apply(getregid)
    orderdf['StartRegionID'] = clustermapdf.ix[orderdf['RegionHash'].values].values[:,0]
    orderdf['Timeslot'] = orderdf[1].apply(totimeslot)
    orderdf['Date'] = orderdf[0].apply(getdate)
    orderdf.drop([0,1],axis=1,inplace=True)

    weatherdf['Timeslot'] = weatherdf['Time'].apply(totimeslot)
    orderdf=pd.merge(orderdf,weatherdf[['Timeslot','Weather','Temperature','PM2.5']],on='Timeslot',how='left')

    temp1=orderdf.groupby(['StartRegionID','Timeslot','Date'])['Weather','Temperature','PM2.5'].mean().reset_index()
    temp2=orderdf.groupby(['StartRegionID','Timeslot','Date'])['Weather'].agg({'Demand':lambda x:int(x.shape[0])}).reset_index()
    orderdf=pd.merge(temp1,temp2,on=['StartRegionID','Timeslot','Date'],how='inner')

    #Finally, adding POI information 
    poi=ConstructPOITable('./training_set/poi_data/poi_data')
    poi['StartRegionID'] = clustermapdf.ix[poi.index.values].values[:,0]
    
    designMatrix=pd.merge(orderdf, poi, on='StartRegionID',how='left')
    
    #Saving the constructed design matrix
    designMatrix.to_csv(outname, sep=',', index=False)
    return designMatrix, orderdf
    
def PredictOnKaggleTestSet(basepath, kagglefname, model, verbose=True, Save=True):
    testfnames = [x for x in sorted(os.listdir(basepath+'DesignMatrices/')) if x[0]!='.']
    clusterfname = basepath+'cluster_map/cluster_map'
    clustermap = pd.read_csv(clusterfname, sep='\t', names=['RegionHash', 'RegionID'], index_col='RegionID')

    KaggleFull = pd.read_csv(kagglefname, index_col='id')
    for fname in testfnames:
        
        if verbose == True:
            print 'predicting for date = {}'.format(fname)
            
        Xtestpd = pd.read_csv(basepath+'DesignMatrices/'+fname)
        Xtestpd['Gap']=model.Predict(Xtestpd)

        KaggleXtest = pd.DataFrame(Xtestpd.Gap.values, columns=['Gap'])
        KaggleXtest['district_date_slot'] = clustermap.ix[list(Xtestpd.StartRegionID.values)].RegionHash.values
        KaggleXtest['district_date_slot'] = KaggleXtest.district_date_slot.astype(str).str.cat(Xtestpd.Date.astype(str), 
                                                                                              sep='_')
        KaggleXtest['district_date_slot'] = KaggleXtest.district_date_slot.astype(str).str.cat(Xtestpd.Timeslot.apply(lambda x:x-1).astype(str), sep='_')
        KaggleFull.ix[KaggleXtest.district_date_slot.values] = KaggleXtest.Gap.values[:, np.newaxis]
    if Save ==True:
        KaggleFull.to_csv(kagglefname)
    else:
        return KaggleFull
    
def MeanAbsoluteError(truths, preds): 
    return np.mean(np.abs(truths-preds))

def one_hot_encode(x, n_classes):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
     """
    return np.eye(n_classes)[x]