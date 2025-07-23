import torch
import torch.nn as nn
import numpy as np
import random,pickle
import matplotlib.pyplot as plt

import nets.pigmentsformer as Mdl
import nets.Dataset_train as ds
import torch.utils.data


def update_lr(optimizer,lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
#         param_group['lr'] = d**(-0.5)*min(epoch**(-0.5),epoch*100**(-1.5))

    
def training(train_x1,train_x2,train_y1,train_y2p,train_y2d,train_y2cal, val_x1,val_x2,val_y1,val_y2p,val_y2d, val_y2cal, test_x1,test_x2,test_y1,test_y2p,test_y2d,test_y2cal, epochs,device,learning_rate,save_best_loss=9999,cv_i=None,param=None):
    ##### initial model
    model = Mdl.TS(train_x1.shape[1],train_x1.shape[2],train_x2.shape[1],train_y1.shape[1],param).to(device)
    # if param["name"] == "LSTM":
    #     model = Mdl.LSTM(train_x1.shape[-1],train_y1.shape[1],train_x2.shape[1]).to(device)
    # elif param["name"] == "RNN":
    #     model = Mdl.RNN(train_x1.shape[-1],train_y1.shape[1],train_x2.shape[1]).to(device)
    # elif param["name"] == "ANN":
    #     model = Mdl.ANN(train_x1.shape[1],train_x1.shape[2],train_x2.shape[1],train_y1.shape[1]).to(device)
    # else:
    #     raise Exception("没有该模型")
        
    criterion = Mdl.my_loss().to(device)
    # criterion_pro = Mdl.loss_pro().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    
    ##### model training
    loss,loss_val,loss_test = [],[],[]
    dataset_train = torch.utils.data.DataLoader(ds.DS(train_x1,train_x2,train_y1,train_y2p,train_y2d,train_y2cal),batch_size=train_x1.shape[0]//2+1,shuffle=True)
    # epoch_bar = tqdm(range(epochs),desc=f"cross validation:{cv_i}")
    for epoch in range(epochs):
        loss_batch = []
        for batchj,(x1,x2,y1,y2p,y2d,y2cal) in enumerate(dataset_train):
            model.train()
            x2[:,-1] = x2[:,-1]*np.random.rand(x2.shape[0])*4./3.+x2[:,-1]/3.
            inputs1 = x1.float().to(device) 
            inputs2 = x2.float().to(device) 
            target = y1.float().to(device) 
            loss_cal_param_p = y2p.float().to(device) 
            loss_cal_param_d = y2d.float().to(device) 
            loss_cal_param_cal = y2cal.float().to(device) 

            outputs = model(inputs1,inputs2)
            loss_value = criterion(outputs,target,loss_cal_param_p,loss_cal_param_d,loss_cal_param_cal)

#             loss_pro = criterion_pro(model,inputs1,inputs2)
#             loss_value = loss_value*(1+loss_pro)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            loss_batch.append(loss_value.item())
        loss.append(np.mean(loss_batch))

        ##### for validation and test
        # for validation, get validation result and chose best model
        with torch.no_grad():
            model.eval()
            inputs1 = torch.tensor(val_x1).float().to(device) 
            inputs2 = torch.tensor(val_x2).float().to(device) 
            target = torch.tensor(val_y1).float().to(device) 
            loss_cal_param_p = torch.tensor(val_y2p).float().to(device) 
            loss_cal_param_d = torch.tensor(val_y2d).float().to(device)  
            loss_cal_param_cal = torch.tensor(val_y2cal).float().to(device) 
        
            outputs = model(inputs1,inputs2)
            loss_val.append(criterion(outputs,target,loss_cal_param_p,loss_cal_param_d,loss_cal_param_cal).item())
            predict_val = outputs.cpu().detach().numpy() 

        # for test, this calculation has no impact for model, only show inner result
        with torch.no_grad():
            model.eval()
            inputs1 = torch.tensor(test_x1).float().to(device) 
            inputs2 = torch.tensor(test_x2).float().to(device) 
            target = torch.tensor(test_y1).float().to(device) 
            loss_cal_param_p = torch.tensor(test_y2p).float().to(device) 
            loss_cal_param_d = torch.tensor(test_y2d).float().to(device) 
            loss_cal_param_cal = torch.tensor(test_y2cal).float().to(device) 

            outputs = model(inputs1,inputs2)
            loss_test.append(criterion(outputs,target,loss_cal_param_p,loss_cal_param_d,loss_cal_param_cal).item())
            predict_test = outputs.cpu().detach().numpy()
        print('[cv_num: {:d}] (epoch: {:d}) train loss: {:.4}, val loss: {:.4}, test loss: {:.4}'.format(cv_i,epoch,loss[-1],loss_val[-1],loss_test[-1]))
            # epoch_bar.set_postfix({'train loss':loss[-1],'val loss':loss_val[-1],'test loss':loss_test[-1]})
    
    # save best result model
    if save_best_loss>loss_val[-1]:
        torch.save({'net':model.state_dict(),'optimizer':optimizer.state_dict()},f"./0 save inner data/best_model-{param['name']}.pth")
    else:
        loss_test,predict_test = None, None
    return loss,loss_val,loss_test,predict_val,predict_test

@ torch.no_grad()
def model_eval(model,inputs1,inputs2,y1_scaler):
    model.eval()
    outputs = model(inputs1,inputs2)
    y1 = y1_scaler.inverse_transform(outputs.cpu().detach().numpy())
    return y1

def main(tobj):
    #logging.basicConfig(filename='./log/training.log',
    #                    format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
    #                    level=logging.INFO)
    # training, using cross-validation
    cv_num = tobj.cv_num 
    epochs = tobj.epochs
    device = tobj.device
    learning_rate = tobj.learning_rate

    test_x1 = tobj.test_x1
    test_x2 = tobj.test_x2
    test_y1 = tobj.test_y1
    test_y2p = tobj.test_y2p
    test_y2d = tobj.test_y2d
    test_y2cal = tobj.test_y2cal
    
    batch_x1s = np.array_split(tobj.x1,cv_num)
    batch_x2s = np.array_split(tobj.x2,cv_num)
    batch_y1s = np.array_split(tobj.y1,cv_num)
    batch_y2ps = np.array_split(tobj.y2p,cv_num)
    batch_y2ds = np.array_split(tobj.y2d,cv_num)
    batch_y2cals = np.array_split(tobj.y2cal,cv_num)

    min_loss,LOSS,predict = 9999, None, []
    for i,(batch_x1,batch_x2,batch_y1,batch_y2p,batch_y2d,batch_y2cal) in enumerate(zip(batch_x1s,batch_x2s,batch_y1s,batch_y2ps,batch_y2ds,batch_y2cals)):
        
        # get train, val
        train_x1,train_x2,train_y1,train_y2p,train_y2d,train_y2cal = [],[],[],[],[],[]
        for j in range(cv_num):
            if i!=j:
                train_x1.append(batch_x1s[j].copy())
                train_x2.append(batch_x2s[j].copy())
                train_y1.append(batch_y1s[j].copy())
                train_y2p.append(batch_y2ps[j].copy())
                train_y2d.append(batch_y2ds[j].copy())
                train_y2cal.append(batch_y2cals[j].copy())
            else:
                val_x1,val_x2,val_y1,val_y2p,val_y2d,val_y2cal = batch_x1.copy(),batch_x2.copy(),batch_y1.copy(),batch_y2p.copy(),batch_y2d.copy(),batch_y2cal.copy()
        train_x1 = np.concatenate(train_x1,axis=0)
        train_x2 = np.concatenate(train_x2,axis=0)
        train_y1 = np.concatenate(train_y1,axis=0)
        train_y2p = np.concatenate(train_y2p,axis=0)
        train_y2d = np.concatenate(train_y2d,axis=0)
        train_y2cal = np.concatenate(train_y2cal,axis=0)
    
        # training
        training_loss,training_val_loss,training_test_loss,val_predict,test_predict = training(train_x1,
                                                                                               train_x2,
                                                                                               train_y1,
                                                                                               train_y2p,
                                                                                               train_y2d,
                                                                                               train_y2cal,
                                                                                               val_x1,
                                                                                               val_x2,
                                                                                               val_y1,
                                                                                               val_y2p,
                                                                                               val_y2d,
                                                                                               val_y2cal,
                                                                                               test_x1,
                                                                                               test_x2,
                                                                                               test_y1,
                                                                                               test_y2p,
                                                                                               test_y2d,
                                                                                               test_y2cal,
                                                                                               epochs=epochs,
                                                                                               device=device,
                                                                                               learning_rate=learning_rate,
                                                                                               save_best_loss=min_loss,
                                                                                               cv_i=i,
                                                                                               param=tobj.param)
        
        if training_test_loss is not None:
            min_loss = training_val_loss[-1]
            test_predict_save_data = test_predict
            LOSS_test = training_test_loss
        
        if LOSS is None:
            LOSS = np.array(training_loss)
            LOSS_val = np.array(training_val_loss)
        else:
            LOSS +=training_loss
            LOSS_val +=training_val_loss

        predict.append(val_predict)

    tobj.LOSS_train = LOSS/cv_num
    tobj.LOSS_val = LOSS_val/cv_num
    tobj.LOSS_test = LOSS_test
    tobj.predict_cv = np.concatenate(predict,axis=0)
    tobj.predict_test = test_predict_save_data

    return tobj


