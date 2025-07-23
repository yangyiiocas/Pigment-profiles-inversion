import numpy as np  
import copy

class MinMaxScaler():
    def __init__(self, x=None):
        self.max = []
        self.min = []
        self.mean = []
        self.isnan = {}
        if x is not None:
            self.feature = x.shape[-1]
            for i in range(self.feature):
                self.max.append(np.nanmax(x[...,i]))
                self.min.append(np.nanmin(x[...,i]))
                self.mean.append(np.nanmean(x[...,i]))

    def transform(self,x0, dataname=None):
        x = copy.deepcopy(x0)
        for i in range(self.feature):
            x[...,i] = (x[...,i]-self.min[i])/(self.max[i]-self.min[i])
        if dataname is not None:
            self.isnan[dataname] = np.isnan(x)
        x[np.isnan(x)] = 0
        return x

    def inverse_transform(self,x0, dataname=None):
        x = copy.deepcopy(x0)
        for i in range(self.feature):
            x[...,i] = x[...,i]*(self.max[i]-self.min[i])+self.min[i]
        
        if dataname is not None:
            x[self.isnan[dataname]] = np.nan
        return x

class StScaler():
    def __init__(self, x=None):
        self.max = []
        self.min = []
        self.mean = []
        self.var = []
        self.isnan = {}
        if x is not None:
            self.feature = x.shape[-1]
            for i in range(self.feature):
                self.max.append(np.nanmax(x[...,i]))
                self.min.append(np.nanmin(x[...,i]))
                self.mean.append(np.nanmean(x[...,i]))
                self.var.append(np.nanvar(x[...,i]))

    def transform(self,x0, dataname=None):
        x = copy.deepcopy(x0)
        for i in range(self.feature):
            x[...,i] = (x[...,i]-self.mean[i])/np.sqrt(self.var[i])
        if dataname is not None:
            self.isnan[dataname] = np.isnan(x)
        x[np.isnan(x)] = 0
        return x

    def inverse_transform(self,x0, dataname=None):
        x = copy.deepcopy(x0)
        for i in range(self.feature):
            x[...,i] = x[...,i]*np.sqrt(self.var[i])+self.mean[i]
        
        if dataname is not None:
            x[self.isnan[dataname]] = np.nan
        return x

def cal_error(output, target):
    output = np.squeeze(output)
    target = np.squeeze(target)
    isnan = np.isnan(output)|np.isnan(target)
    output = output[~isnan]
    target = target[~isnan]
    
    return {"R": np.corrcoef([output,target])[1,0],
            "R2":1-((output-target)**2).sum()/((target-target.mean())**2).sum(),
            "MAE": np.abs(10**output-10**target).mean(),
            "RMSE":np.sqrt(((10**output-10**target)**2).mean()),
            "Bias": (10**output-10**target).mean()*100} 