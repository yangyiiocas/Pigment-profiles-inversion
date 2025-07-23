import numpy as np
import datetime
import pickle

# 提取数据对象
class read_content():
    def __init__(self, file):
        # 文件名，数据长度，变量描述，数据
        self.filename = file.filename.split("/")[-1]
        self.length = file.length
        self.headers = file.headers
        self.variables = {var.lower():file.variables[var] for var in file.variables}
        self.data = {var.lower():file.data[var] for var in file.data}
        try:
            self.time = file.fd_datetime()
        except:
            self.time = None
        self.lat = None 
        self.lon = None 
        self.depth = None 
        
    # 判断是否存在var_use变量
    def is_exist(self, var_use):
        for sub_list in var_use:
            a = True
            for j in sub_list:
                a = a & (j in self.data)
            if a: return True
        return False


    # 获取var_use变量的值
    def get_var(self, var_use):
        for sub_list in var_use:
            a = True
            for j in sub_list:
                a = a & (j in self.data)
            if a:
                return [self.data[j] for j in sub_list]


