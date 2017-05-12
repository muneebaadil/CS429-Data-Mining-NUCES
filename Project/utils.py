from sklearn import linear_model
import numpy as np
import pandas as pd 
import os 

def SplitByRegion(X, supply, demand, splitfrac):
    Xtrain, supplyTrain, demandTrain = [], [], []
    Xval, supplyVal, demandVal = [], [], []
    
    regions = np.unique(X[:, 0])

    for r in regions: 
        Xr, supplyr, demandr = X[X[:, 0]==r], supply[X[:, 0]==r], demand[X[:, 0]==r]
        
        randidx = np.random.choice(Xr.shape[0], Xr.shape[0], replace=False)
        splitidx = int(splitfrac * Xr.shape[0])
        
        Xtrain.append(Xr[randidx[:splitidx]])
        supplyTrain.append(supplyr[randidx[:splitidx]])
        demandTrain.append(demandr[randidx[:splitidx]])
        
        Xval.append(Xr[randidx[splitidx:]])
        supplyVal.append(supplyr[randidx[splitidx:]])
        demandVal.append(demandr[randidx[splitidx:]])
        
    Xtrain, supplyTrain, demandTrain = np.concatenate(Xtrain), np.concatenate(supplyTrain), np.concatenate(demandTrain)
    Xval, supplyVal, demandVal = np.concatenate(Xval), np.concatenate(supplyVal), np.concatenate(demandVal)
    
    return Xtrain, supplyTrain, demandTrain, Xval, supplyVal, demandVal

def LoadDataset(basepath):
    fnames = [x for x in sorted(os.listdir(basepath)) if x[0]!='.']
    #X, supply, demand = None, None, None
    Xs, supplies, demands = [], [], []
    for fname in fnames: 
        table = pd.read_csv(basepath+fname)
        supplies.append(table['Supply'])
        demands.append(table['Demand'])
        Xs.append(table.drop(['Supply', 'Demand'], axis=1))
        
    supply = pd.concat(supplies)
    demand = pd.concat(demands)
    X = pd.concat(Xs)

    return X, supply, demand

totimeslot = lambda x: ((int(x[-5:-3]) + (int(x[-8:-6])*60))//10)+1

def PrepareFeatureVector(setname, ofname, wfname, outname):
    #Defining all names
    weatherfname = setname + 'weather_data/' + wfname
    orderfname = setname + 'order_data/' + ofname
    poifname = setname + 'poi_data/poi_data'
    clustermapfname = setname + 'cluster_map/cluster_map'
    
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
    
    #Constructing a design matrix to store 
    designMatrix = (orderdf.groupby(['StartRegionID', 'Timeslot', 'Weather', 'Temperature', 'PM2.5']))['DriverID'].agg({
                    'Demand': 'count', 'Supply': lambda x:np.sum(x!='NULL')})
    
    designMatrix.to_csv(setname+'DesignMatrices/'+outname, sep=',')
    return designMatrix

getdate = lambda x: x.split(',')[-1]
getregid = lambda x: x.split(',')[2]

def PrepareTestExamples(setname, ofname, wfname, outname):
    #Defining all file names  
    weatherfname = setname + 'weather_data/' + wfname
    orderfname = setname + 'order_data/' + ofname
    poifname = setname + 'poi_data/poi_data'
    clustermapfname = setname + 'cluster_map/cluster_map'

    #Reading all files for the filename given as argument 
    orderdf = pd.read_csv(orderfname, sep=' ', header=None)
    weatherdf = pd.read_csv(weatherfname, sep='\t', header=None, na_filter=False, 
                     names=['time', 'weather', 'temperature', 'pm2.5'])
    clustermapdf = pd.read_csv(clustermapfname, sep='\t', 
                               names=['RegionHash', 'RegionID'], index_col = 0)
    
    orderdf['Timeslot'] = orderdf[1].apply(totimeslot)
    orderdf['RegionHash'] = orderdf[0].apply(getregid)
    orderdf['Date'] = orderdf[0].apply(getdate)
    orderdf['RegionID'] = clustermapdf.ix[orderdf['RegionHash'].values].values[:,0]

    weatherdf['timeslot'] = weatherdf['time'].apply(totimeslot)

    orderdf['Weather'] = weatherdf.groupby('timeslot')['weather'].mean().ix[orderdf['Timeslot'].values].values
    orderdf['Temperature'] = (weatherdf.groupby('timeslot')['temperature'].mean()).ix[orderdf['Timeslot'].values].values
    orderdf['PM2.5'] = (weatherdf.groupby('timeslot')['pm2.5'].mean()).ix[orderdf['Timeslot'].values].values
    
    orderdf = orderdf.drop([0, 1], axis=1)

    orderdf.to_csv(setname+'DesignMatrices/'+outname, sep=',')
    return orderdf

def WriteOnKaggleFormat(Xtestpd, Ytest, date):
    myout = pd.DataFrame(Ytest, columns=['gap'])
    myout['district_date_slot'] = Xtestpd['RegionHash'].values
    myout['date'] = date
    #myout
    myout['district_date_slot'] = myout.district_date_slot.astype(str).str.cat(Xtestpd.Date.astype(str), sep='_')
    myout['district_date_slot'] = myout.district_date_slot.astype(str).str.cat(Xtestpd.Timeslot.astype(str), sep='-')
    #myout=myout.set_index('district_date_slot')
    
    kaggleout = pd.read_csv('sample.csv', index_col='district_date_slot')
    kaggleout.ix[myout['district_date_slot'].values] = myout['gap'].values[:, np.newaxis]
    #kaggleout.loc[myout['district_date_slot'].values].gap = myout.gap.values
    kaggleout.to_csv('sample.csv')
    
def MeanAbsoluteError(truths, preds): 
    return np.mean(np.abs(truths-preds))

def CreateWeatherVects(foldername, outfolder):
    fnames = [x for x in sorted(os.listdir(foldername)) if x[0]!='.']
    for fname in fnames: 
        wdf = pd.read_csv(foldername+fname, sep='\t', 
                          names=['Timestamp', 'Weather','Temperature', 'PM2.5'], header=None)
        wdf['Timeslot']=wdf.Timestamp.apply(utils.totimeslot)
        print fname 
        wdf.groupby('Timeslot').Weather.mean().astype(int).to_csv(outfolder+'wmatrix_'+fname.split('_')[2]+'.csv')
    return

def
    