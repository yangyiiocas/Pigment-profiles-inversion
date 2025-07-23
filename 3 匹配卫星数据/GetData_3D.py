import os
import numpy as np
import netCDF4 as nc
import h5py
import scipy as scio
import datetime
import pickle




class DataObj():
    def __init__(self, date_name, window_t, window_s, variables_param, FilePathNameFind_Fun):
        self.date_name = date_name
        self.win_s     = window_s
        self.win_t     = window_t
        self.path      = variables_param["datafilepath"]
        self.var       = variables_param["var"]
        self.lon_var   = variables_param["lon_var"]
        self.lat_var   = variables_param["lat_var"]
        self.depth_var = variables_param["depth_var"]
        self.pathfile = None

        y,m,d = int(self.date_name[:4]),int(self.date_name[4:6]),int(self.date_name[6:])
        yd = (datetime.datetime(y,m,d)-datetime.datetime(y,1,1)).days+1
        self.pathfile = FilePathNameFind_Fun(self.path,y,m,d,yd)


        # read data file and load latter
        self.i         = []
        self.win_index = []
        self.lat       = []
        self.lon       = []
        self.date      = []
        self.depth     = []

        
    def add_situ(self, i, win_index, lat, lon, date, depth):
        self.i.append(i)
        self.win_index.append(win_index)
        self.lat.append(lat)
        self.lon.append(lon)
        self.date.append(date)
        self.depth.append(depth)
        
        
    def add_data_space(self):
        self.length = len(self.lat)
        self.data = np.nan*np.zeros([self.length, self.win_s, self.win_s])
        self.lat_center = np.nan*np.zeros(self.length)
        self.lon_center = np.nan*np.zeros(self.length)
        self.depth_inter = np.nan*np.zeros(self.length)
    

    def add_data(self):
        if self.pathfile is None: return

        data_file = self.pathfile
        try:
            with nc.Dataset(data_file) as f:    
                lon_ext, lat_ext, depth_ext = f[self.lon_var][:], f[self.lat_var][:], f[self.depth_var][:]
                var_ext = np.squeeze(f[self.var][:])
            var_ext.data[var_ext.mask == True] = np.nan
        except:
            # print("\n",data_file,end="\n")
            return  
        var_ext = var_ext.data  


        for i in range(self.length): 
            a, b, w = np.argmin(abs(lat_ext-self.lat[i])), np.argmin(abs(lon_ext-self.lon[i])), (self.win_s-1)//2
            c = np.arange(b-w,b+w+1) % len(lon_ext)
            d = np.argmin(abs(depth_ext-self.depth[i]))

            self.lat_center[i] = lat_ext[a]
            self.lon_center[i] = lon_ext[b]
            self.depth_inter[i] = depth_ext[d]

            extract = np.squeeze(var_ext[d,a-w:a+w+1,c])
            if extract.shape==(self.win_s,self.win_s):
                self.data[i,:,:] =  extract
            else:
                d = extract.shape[0]
                if self.lat[i]>0: self.data[i,-d:,:] = extract
                else: self.data[i,:d,:] = extract




def read_data(dates, lon, lat, depth, window_t,window_s,variables_param, FilePathNameFind_Fun):
    print("use length:{:d}".format(len(dates)))
    
    # 需要使用的天数据记录下来，这样可以只读取一次
    data_obj = {}
    for i in range(len(dates)):
        dates_win = [dates[i]+datetime.timedelta(days=t) for t in range(-(window_t-1)//2,(window_t-1)//2+1)]
        for win_index in range(len(dates_win)):
            date_name = "{:0>4d}{:0>2d}{:0>2d}".format(dates_win[win_index].year,
                                                       dates_win[win_index].month,
                                                       dates_win[win_index].day)
            
            if date_name not in data_obj:
                data_obj[date_name] = DataObj(date_name, window_t, window_s, variables_param, FilePathNameFind_Fun)
            data_obj[date_name].add_situ(i=i, 
                                         win_index=win_index, 
                                         lat=lat[i], 
                                         lon=lon[i],
                                         date=dates[i],
                                         depth=depth[i])
    print("all data file:{:d}".format(len(data_obj)))
    
    # 读取satellite数据
    m = len(data_obj)
    for i,obj in enumerate(data_obj):
        data_obj[obj].add_data_space()
        data_obj[obj].add_data() 
        print("read inforamtion, number:{:d}, percentage process:{:.4%}, all data file:{:d}".format(i+1, (i+1)/m, m), end="\r")
    print("\n read inforamtion done!")

    # 将读取的数据转化为对应的结果
    m, count = len(dates), 0
    data = np.nan*np.zeros([m,window_t,window_s,window_s])

    for j,obj in enumerate(data_obj):
        if data_obj[obj].pathfile is not None:
            data[data_obj[obj].i,data_obj[obj].win_index,:,:] = data_obj[obj].data

        else:
            data[data_obj[obj].i,data_obj[obj].win_index,:,:] = np.nan*np.zeros([data_obj[obj].length,
                                                                                 data_obj[obj].win_s,
                                                                                 data_obj[obj].win_s])
        count = count + data_obj[obj].length
        print("write data inforamtion,  number: {:d},  percentage: {:.4%}".format(count, count/m/window_t), end="\r")
    print("\n write data inforamtion done!")
    
    return data

