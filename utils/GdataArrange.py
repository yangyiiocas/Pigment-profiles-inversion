import os,datetime
import numpy as np

from utils.RSData import ReadSatelliteData_simple as RS
from utils.ParamCal import strictcase1water
depths = [0.494025,1.541375,2.645669,3.819495,5.078224,6.440614,7.92956,
          9.572997,11.405,13.46714,15.81007,18.49556,21.59882,25.21141,
          29.44473,34.43415,40.34405,47.37369,55.76429,65.80727,77.85385,
          92.32607,109.7293,130.666,155.8507,186.1256,222.4752,266.0403]

def ChoseData(x):
    if len(x.shape)!=3:
        raise ValueError("not match dim!")
    use1 = (np.isnan(x).sum(axis=2)<=8).all(axis=1)
    use2 = (~np.isnan(x)).any(axis=-1).any(axis=-1)
    use3 = strictcase1water(Rrs412=np.nanmean(10**x[:,:,6],axis=-1),
                            Rrs443=np.nanmean(10**x[:,:,7],axis=-1),
                            Rrs490=np.nanmean(10**x[:,:,8],axis=-1),
                            Rrs560=np.nanmean(10**x[:,:,10],axis=-1))
    return use1&use2&use3

class Dataset():
    def __init__(self, std):
        self.thistime = std
        self.ds = []
        self.svari = ["tem","so","uo","vo","ssh","log10_mld","log10_Rrs412","log10_Rrs443","log10_Rrs490","log10_Rrs510","log10_Rrs560","log10_Rrs665","chla","Zeu"]
        self.pvari = ["fuco","perid","but_fuco","hex_fuco","allo","tchlb","zea","tchla","dvchla"]
        self.lat = np.arange(-89.5,90,1)
        self.lon = np.arange(-179.5,180,1)
        self.depth = np.array(depths)
        for dt in range(-7,8):
            date = std+datetime.timedelta(dt)
            y,m,d = date.year,date.month,date.day
            data = RS(y,m,d)
            data = np.concatenate([d.flatten()[:,np.newaxis,np.newaxis] for d in data],axis=2)
            self.ds.append(data)
        self.N = self.ds[0].shape[0]
        self.use = ChoseData(np.array(np.concatenate(self.ds.copy(), axis=1)))
    # update the ds and usedata
    def nextdate(self, date):
        self.thistime = date
        self.ds.pop(0)
        y,m,d = date.year,date.month,date.day
        data = RS(y,m,d)
        data = np.concatenate([d.flatten()[:,np.newaxis,np.newaxis] for d in data],axis=2)
        self.ds.append(data)
        self.use = ChoseData(np.array(np.concatenate(self.ds.copy(), axis=1)))
    # get N*C*L data and all available
    def getx2(self,):
        gdepth,glat,glon = np.meshgrid(self.depth, self.lat, self.lon, indexing='ij')
        return np.array([np.ones(self.N,dtype=float)*np.sin(self.thistime.month/6.*np.pi),
                     np.ones(self.N,dtype=float)*np.cos(self.thistime.month/6.*np.pi),
                     np.ones(self.N,dtype=float)*np.sin(self.thistime.day/31.*2*np.pi),
                     np.ones(self.N,dtype=float)*np.cos(self.thistime.day/31.*2*np.pi),
                     np.sin(glon.flatten()/180.*np.pi),
                     np.cos(glon.flatten()/180.*np.pi),
                     glat.flatten()/90.,
                     gdepth.flatten()/300.]).transpose(1,0)
    def getdata(self, date):
        if date!=self.thistime:
            if (date-self.thistime).days!=1:
                raise ValueError("时间不连续")
            else:
                self.nextdate(date)
        return np.array(np.concatenate(self.ds.copy(), axis=1))[self.use], self.getx2()[self.use]
    # restruct global data
    def ReGlobal(self, mdlout):
        Lx = mdlout.shape[-1]
        out = np.zeros((self.N,Lx))*np.nan
        out[self.use] = mdlout
        return out.reshape(len(self.depth),len(self.lat),len(self.lon),Lx)