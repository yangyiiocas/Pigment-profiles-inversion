import torch
import torch.nn as nn
import torch.nn.functional as F 

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class my_loss(nn.Module):
    def __init__(self,):
        super(my_loss, self).__init__()
        self.RMSE = nn.MSELoss(reduction='sum')
        self.MAE = nn.L1Loss(reduction='sum')
        self.beta1 = 1.0
        self.beta2 = 0.0
        self.beta3 = 0.0
    def forward(self, model_outputs, y1, y2p, y2d, y2cal):
        MAE = self.MAE(model_outputs*y2cal, y1*y2cal)/y2cal.sum()
        return MAE

def ff(in_num,out_num,hidden):
    inner_num = 64
    if hidden == 0:
        return nn.Linear(in_num,out_num)
    layer = [nn.Linear(in_num,inner_num),nn.PReLU()]
    for i in range(hidden-1):
        layer.append(nn.Linear(inner_num,inner_num))
        layer.append(nn.PReLU())
    layer.append(nn.Linear(inner_num,out_num))
    return nn.Sequential(*layer)


class mh_attn(nn.Module):
    def __init__(self, d_model,heads):
        super(mh_attn, self).__init__()
        self.fn = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)]) # 使用了nn.MultiheadAttention， 无需Wo
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=0.0, batch_first=True)
    def forward(self, query, key=None, value=None):
        if key is None:
            key = query
        if value is None:
            value = key
        query, key, value = [fni(x) for (fni,x) in zip(self.fn, (query, key, value))]
        x = self.attn(query, key, value)[0]
        return x

class Encoder(nn.Module):
    def __init__(self,seq_len,d_model,heads,dim_k,dim_v):
        super(Encoder, self).__init__()
        self.seq_len = seq_len 
        self.d_model = d_model
        self.head = heads

        self.attn1 = mh_attn(d_model,heads)
        self.bn1 = nn.LayerNorm(normalized_shape=d_model)
        self.ff = ff(d_model,d_model,1)
        self.bn2 = nn.LayerNorm(normalized_shape=d_model)

    def forward(self,x):
        x = x + self.attn1(x.view(-1,self.seq_len,self.d_model))
        x = self.bn1(x)
        x = x + self.ff(x.view(-1,self.d_model)).view(-1,self.seq_len,self.d_model)
        return self.bn2(x)


class Decoder(nn.Module):
    def __init__(self,seq_len,d_model,heads,dim_k,dim_v):
        super(Decoder, self).__init__()
        self.seq_len = seq_len 
        self.d_model = d_model
        self.head = heads

        self.attn1 = mh_attn(d_model,heads)
        self.bn1 = nn.LayerNorm(normalized_shape=d_model)
        self.attn2 = mh_attn(d_model,heads)
        self.bn2 = nn.LayerNorm(normalized_shape=d_model)
        self.ff = ff(d_model,d_model,3)

    def forward(self,xq,xkv):
        x = xq + self.attn1(xq.view(-1,self.seq_len,self.d_model))
        x = self.bn1(x)
        x = x + self.attn2(x.view(-1,self.seq_len,self.d_model),xkv.view(-1,self.seq_len,self.d_model))
        x = self.bn2(x)
        x = x + self.ff(x.view(-1,self.d_model)).view(-1,self.seq_len,self.d_model)
        return x


class TS(nn.Module):
    def __init__(self, Lx, Lst, out_num, **param):
        super(TS, self).__init__()
        encoder_num = param["encoder_num"]
        decoder_num = param["decoder_num"]
        d_model = param["d_model"]
        seq_len = param["seq_len"]
        heads = param["heads"]
        dim_k = param["dim_k"]
        dim_v = param["dim_v"]

        self.fn_embed = nn.Linear(Lx, d_model)
        self.activate = nn.PReLU()
        self.PE = (torch.arange(-(seq_len-1)/2.,(seq_len-1)/2.+1,1)/seq_len*torch.pi).cos_().unsqueeze(dim=1).repeat(1,d_model)

        self.encoders = nn.ModuleList(Encoder(seq_len,d_model,heads,dim_k,dim_v) for i in range(encoder_num))
        self.decoders = nn.ModuleList(Decoder(seq_len,d_model,heads,dim_k,dim_v) for i in range(decoder_num))
        self.fn_st = nn.Linear(Lst,d_model)
        self.fn_out = nn.Linear(seq_len*d_model,out_num)

    def forward(self, x, st):
        
        x_kv = self.fn_embed(x)
        N,seq_len,d_model = x_kv.size()

        pe = self.PE.repeat(N,1,1).to(x_kv.device)
        x_kv = x_kv + pe
        for encoder in self.encoders:
            x_kv = encoder(x_kv)
        
        x1 = pe + self.fn_st(st).unsqueeze(dim=1).repeat(1,seq_len,1)
        for decoder in self.decoders:
            x1 = decoder(x1,x_kv)
            
        return self.fn_out(x1.view(N,-1))
    



    