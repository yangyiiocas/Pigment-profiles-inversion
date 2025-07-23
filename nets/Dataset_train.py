import torch.utils.data

class DS(torch.utils.data.Dataset):
    def __init__(self, x1,x2,y1,y2p,y2d,y2cal):
        super(DS, self).__init__()
        self.x1 = x1 
        self.x2 = x2 
        self.y1 = y1  
        self.y2p = y2p  
        self.y2d = y2d  
        self.y2cal = y2cal 

    def __getitem__(self, index):
        return self.x1[index],self.x2[index],self.y1[index],self.y2p[index],self.y2d[index],self.y2cal[index]

    def __len__(self,):
        return self.x1.shape[0]

