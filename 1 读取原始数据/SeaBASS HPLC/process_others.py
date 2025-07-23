import numpy as np 
import datetime
# 如果给出的程序无法读取时间，则利用该方法读取
def revise_time(read_obj, var_use):
    t = {"year":np.nan*np.zeros(read_obj.length),
        "month":np.nan*np.zeros(read_obj.length),
        "day":np.nan*np.zeros(read_obj.length),
        "hour":np.nan*np.zeros(read_obj.length),
        "minute":np.nan*np.zeros(read_obj.length),
        "second":np.nan*np.zeros(read_obj.length),
    }
    for sub_list in var_use:
        if sub_list in read_obj.data:
            tmps = read_obj.data[sub_list]
            if (sub_list!="date") & (sub_list!="time") & (sub_list!="sdy"):
                for i in range(read_obj.length): t[sub_list][i] = tmps[i]
            if sub_list == "date":
                for i in range(read_obj.length):
                    t["year"][i] = int(str(tmps[i])[:4])
                    t["month"][i] = int(str(tmps[i])[4:6])
                    t["day"][i] = int(str(tmps[i])[6:8])
            if sub_list == "time":
                for i in range(read_obj.length):
                    if len(str(tmps[i]).split(":"))==3:
                        t["hour"][i] = int(str(tmps[i]).split(":")[0])
                        t["minute"][i] = int(str(tmps[i]).split(":")[1])
                        t["second"][i] = int(str(tmps[i]).split(":")[2])
                    else:
                        t["hour"][i],t["minute"][i],t["second"][i] = np.nan, np.nan, np.nan
            if sub_list == "sdy":
                for i in range(read_obj.length):
                    m_d = datetime.datetime(t["year"][i],1,1)+datetime.timedelta(days=int(tmps[i])-1)
                    t["month"][i] = (m_d).month
                    t["day"][i] = (m_d).day
    
    if np.isnan(t["year"]+t["month"]+t["day"]).sum()!=0:
        start_date = read_obj.headers["start_date"]
        end_date = read_obj.headers["end_date"]
        if len(start_date)==8&len(end_date)==8:
            start_date_year = int(start_date[:4])
            end_date_year = int(end_date[:4])
            start_date_month = int(start_date[4:6])
            end_date_month = int(end_date[4:6])
            start_date_day = int(start_date[6:])
            end_date_day = int(end_date[6:])
            t["year"][np.isnan(t["year"])] = (start_date_year+end_date_year)//2
            t["month"][np.isnan(t["month"])] = (start_date_month+end_date_month)//2
            t["day"][np.isnan(t["day"])] = (start_date_day+end_date_day)//2

    if np.isnan(t["hour"]+t["minute"]+t["second"]).sum()!=0:
        start_time = read_obj.headers["start_time"].replace("[GMT]","").split(":")
        end_time = read_obj.headers["end_time"].replace("[GMT]","").split(":")
        if len(start_time)==3&len(end_time)==3:
            start_time_hour = int(start_time[0])
            end_time_hour = int(end_time[0])
            start_time_minute = int(start_time[1])
            end_time_minute = int(end_time[1])
            start_time_second = int(start_time[2])
            end_time_second = int(end_time[2])
            t["hour"][np.isnan(t["hour"])] = (start_time_hour+end_time_hour)//2
            t["minute"][np.isnan(t["minute"])] = (start_time_minute+end_time_minute)//2
            t["second"][np.isnan(t["second"])] = (start_time_second+end_time_second)//2

    for var in ["hour","minute","second"]: t[var][np.isnan(t[var])] = 0
    for i,s in enumerate(t["second"]): 
        if s>=60: 
            t["second"][i] -=60
            t["minute"][i] +=1
    for var in t: t[var] = t[var].astype(int)
    return [datetime.datetime(t["year"][i],t["month"][i],t["day"][i],t["hour"][i],t["minute"][i],t["second"][i]) for i in range(read_obj.length)]

   

# 该方法读取经纬度
def revise_lat(read_obj,var_use):
    # 从data数据中读取

    if read_obj.is_exist(var_use):
        lat = np.double(read_obj.get_var(var_use))
    else:
        lat = np.nan*np.zeros(np.sum(read_obj.length,axis=0)) 

    if np.isnan(lat).sum()==0:   return lat

    north_latitude = float(read_obj.headers["north_latitude"].replace("[DEG]",""))
    south_latitude = float(read_obj.headers["south_latitude"].replace("[DEG]",""))
    tmp = np.linspace(south_latitude,north_latitude,read_obj.length)
    lat[np.isnan(lat)] = tmp[np.isnan(lat)]
    # test, the difference of two location
    # if abs(south_latitude-north_latitude)>0.25: print(read_obj.filename.split("/")[-1])
    return lat



def revise_lon(read_obj,var_use):
    # 从data数据中读取
    if read_obj.is_exist(var_use):
        lon = np.double(np.sum(read_obj.get_var(var_use),axis=0))
    else:
        lon = np.nan*np.zeros(read_obj.length) 

    if np.isnan(lon).sum()==0:   return lon

    west_longitude = float(read_obj.headers["west_longitude"].replace("[DEG]",""))
    east_longitude = float(read_obj.headers["east_longitude"].replace("[DEG]",""))
    tmp = np.linspace(west_longitude,east_longitude,read_obj.length)
    lon[np.isnan(lon)] = tmp[np.isnan(lon)]
    # if abs(west_longitude-east_longitude)>0.25: print(read_obj.filename.split("/")[-1])
    return lon



# 利用该方法读取depth
def revise_depth(read_obj,var_use):
    # 从data数据中读取
    if read_obj.is_exist(var_use):
        depth = np.double(np.sum(read_obj.get_var(var_use),axis=0))
    else:
        depth = np.nan*np.zeros(read_obj.length) 

    if np.isnan(depth).sum()==0:   return depth
    measurement_depth = float(read_obj.headers["measurement_depth"])
    depth[np.isnan(depth)] = measurement_depth

    return depth