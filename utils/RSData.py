import os,pickle,datetime
import netCDF4 as nc
import torch
import torch.nn.functional as F 
import numpy as np
# import matplotlib.pyplot as plt
from utils import ParamCal

def FilePathNameFind_tsuvm(path, y,m,d):
    filename = None
    wpath = path + "{:0>4d}/{:0>2d}/".format(y,m)
    for file in os.listdir(wpath):
        if "mercatorglorys12v1_gl12_mean_{:0>4d}{:0>2d}{:0>2d}".format(y,m,d) in file:
            filename = wpath + file  
    return filename
def FilePathNameFind_Rrs(path, y,m,d):
    filename = path+"{:0>4d}/ESACCI-OC-L3S-RRS-MERGED-1D_DAILY_4km_GEO_PML_RRS-{:0>4d}{:0>2d}{:0>2d}-fv5.0.nc".format(y,y,m,d)
    return filename if os.path.exists(filename) else None
def FilePathNameFind_Chla(path, y,m,d):
    filename = path+"{:0>4d}/ESACCI-OC-L3S-CHLOR_A-MERGED-1D_DAILY_4km_GEO_PML_OCx-{:0>4d}{:0>2d}{:0>2d}-fv5.0.nc".format(y,y,m,d)
    return filename if os.path.exists(filename) else None


def read_nc_tsuv(y,m,d, vari, pathfile):
    with nc.Dataset(pathfile) as f:
        tmp = f[vari][:,:28,:,:]
    tmp.data[tmp.mask == True] = np.nan
    rdata = tmp.data
    ava_num = (1-np.isnan(rdata)).astype(float)
    rdata[np.isnan(rdata)] = 0
    ava_num = torch.tensor(ava_num).float()
    ava_num = F.avg_pool2d(ava_num, 12, divisor_override=1)
    rdata = torch.tensor(rdata).float()
    rdata = F.avg_pool2d(rdata, 12, divisor_override=1)
    rdata = (rdata/ava_num).numpy()
    ava_num = ava_num.numpy()
    rdata[ava_num<12*12/2] = np.nan
    rdata = np.concatenate([np.nan*np.zeros((1,28,10,360)),rdata],axis=2)
    return rdata 
def read_nc_ms(y,m,d, vari, pathfile):
    with nc.Dataset(pathfile) as f:
        tmp = f[vari][:]
    tmp.data[tmp.mask == True] = np.nan
    rdata = tmp.data[np.newaxis,:,:,:]
    ava_num = (1-np.isnan(rdata)).astype(float)
    rdata[np.isnan(rdata)] = 0
    ava_num = torch.tensor(ava_num).float()
    ava_num = F.avg_pool2d(ava_num, 12, divisor_override=1)
    rdata = torch.tensor(rdata).float()
    rdata = F.avg_pool2d(rdata, 12, divisor_override=1)
    rdata = (rdata/ava_num).numpy()
    ava_num = ava_num.numpy()
    rdata[ava_num<0.33] = np.nan
    rdata = np.concatenate([np.nan*np.zeros((1,28,10,360)),rdata.repeat(28,axis=1)],axis=2)
    return rdata
def read_nc_rc(y,m,d,vari, pathfile):
    with nc.Dataset(pathfile) as f:
        tmp = f[vari][:]
    tmp.data[tmp.mask == True] = np.nan
    rdata = tmp.data[np.newaxis,:,:,:]
    ava_num = (1-np.isnan(rdata)).astype(float)
    rdata[np.isnan(rdata)] = 0
    ava_num = torch.tensor(ava_num).float()
    ava_num = F.avg_pool2d(ava_num, 24, divisor_override=1)
    rdata = torch.tensor(rdata).float()
    rdata = F.avg_pool2d(rdata, 24, divisor_override=1)
    rdata = (rdata/ava_num).numpy()
    ava_num = ava_num.numpy()
    rdata[ava_num<0.33] = np.nan
    return rdata[:,:,-1::-1,:].repeat(28,axis=1) # lat坐标是反着的90~-90，调整过来
def ReadSatelliteData(y,m,d):
    # T S U V
    tem = read_nc_tsuv(y,m,d,"thetao",FilePathNameFind_tsuvm("F:/GLORYS12V1/",y,m,d))
    so = read_nc_tsuv(y,m,d,"so",FilePathNameFind_tsuvm("F:/GLORYS12V1/",y,m,d))
    uo = read_nc_tsuv(y,m,d,"uo",FilePathNameFind_tsuvm("F:/GLORYS12V1/",y,m,d))
    vo = read_nc_tsuv(y,m,d,"vo",FilePathNameFind_tsuvm("F:/GLORYS12V1/",y,m,d))
    # MLD SSH
    mld = read_nc_ms(y,m,d, "mlotst", FilePathNameFind_tsuvm("F:/GLORYS12V1/",y,m,d))
    ssh = read_nc_ms(y,m,d, "zos", FilePathNameFind_tsuvm("F:/GLORYS12V1/",y,m,d))
    # Rrs    
    Rrs412 = read_nc_rc(y,m,d,"Rrs_412",FilePathNameFind_Rrs("G:/occci-v5.0/Rrs/",y,m,d))
    Rrs443 = read_nc_rc(y,m,d,"Rrs_443",FilePathNameFind_Rrs("G:/occci-v5.0/Rrs/",y,m,d))
    Rrs490 = read_nc_rc(y,m,d,"Rrs_490",FilePathNameFind_Rrs("G:/occci-v5.0/Rrs/",y,m,d))
    Rrs510 = read_nc_rc(y,m,d,"Rrs_510",FilePathNameFind_Rrs("G:/occci-v5.0/Rrs/",y,m,d))
    Rrs560 = read_nc_rc(y,m,d,"Rrs_560",FilePathNameFind_Rrs("G:/occci-v5.0/Rrs/",y,m,d))
    Rrs665 = read_nc_rc(y,m,d,"Rrs_665",FilePathNameFind_Rrs("G:/occci-v5.0/Rrs/",y,m,d))
    chla = read_nc_rc(y,m,d,"chlor_a",FilePathNameFind_Chla("K:/occci-v5.0/chlor_a/",y,m,d))
    Zeu = ParamCal.Zeu(chla,option="Chl")
    return [tem,so,uo,vo,
            ssh,np.log10(mld),
            np.log10(Rrs412+10**-6),np.log10(Rrs443+10**-6),
            np.log10(Rrs490+10**-6),np.log10(Rrs510+10**-6),
            np.log10(Rrs560+10**-6),np.log10(Rrs665+10**-6),
            np.log10(chla+10**-6), Zeu]
def ReadSatelliteData_simple(y,m,d):
    file = "/sharedata/uyangyi211136/XDA19060104Private/RS/RS_{:0>4d}{:0>2d}{:0>2d}.pkl".format(y,m,d)
    if not os.path.exists(file):
        raise ValueError
    with open(file,'rb') as f:
        rs = pickle.load(f)
    return rs

