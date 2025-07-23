import torch
import torch.nn as nn
import numpy as np
import os,shutil,datetime,pickle,random,argparse

import scipy.io as scio
import matplotlib.pyplot as plt

import nets.train as train
import utils.data_process as data_process
import nets.train_obj as train_obj

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='model training parameter')

    parser.add_argument('-data', '--data_file', type=str, help='data file name')
    parser.add_argument('-n', '--name', type=str, help='model name')

    parser.add_argument('-d_model','--d_model', type=int, default=32, help='d_model, the last dim')
    parser.add_argument('-seq','--seq_len', type=int, default=15, help='seq length, the inner dim')
    parser.add_argument('-dk', '--dim_k', type=int, default=64, help='dim_k num')
    parser.add_argument('-dv', '--dim_v', type=int, default=64, help='dim_v num')
    parser.add_argument('-heads', '--heads', type=int, default=4, help='heads num')
    parser.add_argument('-eN', '--encoder_num', type=int, default=6, help='encoder_num num')
    parser.add_argument('-dN', '--decoder_num', type=int, default=6, help='decoder_num num')

    parser.add_argument('-lr', '--learning_rate', type=float, default=0.00002, help='learning rate')
    parser.add_argument('-epochs', '--epochs', type=int, default=1500, help='epochs')

    args = parser.parse_args()

    print("data_file:", args.data_file)
    print("model name:", args.name)
    print("d_model:", args.d_model)
    print("seq length:", args.seq_len)
    print("dim_k num:", args.dim_k)
    print("dim_v num:", args.dim_v)
    print("heads num:", args.heads)
    print("encoder_num num:", args.encoder_num)
    print("decoder_num num:", args.decoder_num)
    print("learning rate:", args.learning_rate)
    print("epochs:", args.epochs)



    ############################################################################################################
    mdl_param = {"name" : args.name,
                 "d_model" : args.d_model,
                 "seq_len" : args.seq_len,
                 "dim_k" : args.dim_k,
                 "dim_v" : args.dim_v,
                 "heads": args.heads,
                 "encoder_num":args.encoder_num,
                 "decoder_num":args.decoder_num}

    tobj = train_obj.training_object(param=mdl_param,
                                     device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                                     learning_rate=args.learning_rate,
                                     epochs=args.epochs,
                                     cv_num=10)

    # read data
    with open("./0 save inner data/"+args.data_file,'rb') as f:
        x1,x2,y1,y2p,y2d,y2cal,source = pickle.load(f)
    print(x1.shape,x2.shape,y1.shape,y2p.shape,y2d.shape,y2cal.shape)

    # reshape data
    x1 = np.ascontiguousarray(x1.transpose(0,2,1))
    print(x1.shape,x1.data.contiguous)

    # random data order
    idx_rand = random.sample(range(x1.shape[0]),x1.shape[0])
    x1 = x1[idx_rand]
    x2 = x2[idx_rand]
    y1 = y1[idx_rand]
    y2p = y2p[idx_rand]
    y2d = y2d[idx_rand]
    y2cal = y2cal[idx_rand]
    source = np.array(source,dtype=object)[idx_rand]

    # split dataset
    test_dataset_choose = np.array([s==["Kramer sorted"] for s in source])

    x1_train_val,x1_test = x1[~test_dataset_choose],x1[test_dataset_choose]
    x2_train_val,x2_test = x2[~test_dataset_choose],x2[test_dataset_choose]
    y1_train_val,y1_test = y1[~test_dataset_choose],y1[test_dataset_choose]
    y2p_train_val,y2p_test = y2p[~test_dataset_choose],y2p[test_dataset_choose]
    y2d_train_val,y2d_test = y2d[~test_dataset_choose],y2d[test_dataset_choose]
    y2cal_train_val,y2cal_test = y2cal[~test_dataset_choose],y2cal[test_dataset_choose]

    idx_rand = random.sample(range(x1_train_val.shape[0]),x1_train_val.shape[0])
    x1_train_val = x1_train_val[idx_rand]
    x2_train_val = x2_train_val[idx_rand]
    y1_train_val = y1_train_val[idx_rand]
    y2p_train_val = y2p_train_val[idx_rand]
    y2d_train_val = y2d_train_val[idx_rand]
    y2cal_train_val = y2cal_train_val[idx_rand]

    print("train&val data shape: ",x1_train_val.shape,x2_train_val.shape,y1_train_val.shape,y2p_train_val.shape,y2d_train_val.shape,y2cal_train_val.shape,end="\n\n")
    print("test data shape:",x1_test.shape,x2_test.shape,y1_test.shape,y2p_test.shape,y2d_test.shape,y2cal_test.shape)

    x1_scaler = data_process.MinMaxScaler(x1_train_val)
    y1_scaler = data_process.MinMaxScaler(y1_train_val)

    # standard data, norm
    std_x1_train_val = x1_scaler.transform(x1_train_val,'train_val')
    std_x2_train_val = x2_train_val
    std_y1_train_val = y1_scaler.transform(y1_train_val,'train_val')
    std_y2p_train_val = y2p_train_val
    std_y2d_train_val = y2d_train_val
    std_y2cal_train_val = y2cal_train_val

    std_x1_test = x1_scaler.transform(x1_test,'test')
    std_x2_test = x2_test
    std_y1_test = y1_scaler.transform(y1_test,'test')
    std_y2p_test = y2p_test
    std_y2d_test = y2d_test
    std_y2cal_test = y2cal_test
    print("train&val data shape: ",std_x1_train_val.shape,std_x2_train_val.shape,std_y1_train_val.shape,std_y2p_train_val.shape,std_y2d_train_val.shape,std_y2cal_train_val.shape,end="\n\n")
    print("test data shape:",std_x1_test.shape,std_x2_test.shape,std_y1_test.shape,std_y2p_test.shape,std_y2d_test.shape,std_y2cal_test.shape,end="\n\n")

    # load data
    tobj.idx_rand = idx_rand
    tobj.test_dataset_choose = test_dataset_choose

    tobj.x1 = std_x1_train_val
    tobj.x2 = std_x2_train_val
    tobj.y1 = std_y1_train_val
    tobj.y2p = std_y2p_train_val
    tobj.y2d = std_y2d_train_val
    tobj.y2cal = std_y2cal_train_val

    tobj.test_x1 = std_x1_test
    tobj.test_x2 = std_x2_test
    tobj.test_y1 = std_y1_test
    tobj.test_y2p = std_y2p_test
    tobj.test_y2d = std_y2d_test
    tobj.test_y2cal = std_y2cal_test

    tobj.x1_scaler = x1_scaler
    tobj.y1_scaler = y1_scaler


    # cross validation: training
    print("training begin! now!!!!!!!")
    train.main(tobj)
    print("training done!")
    LOSS_train = tobj.LOSS_train
    LOSS_val = tobj.LOSS_val
    LOSS_test = tobj.LOSS_test
    predict_cv = tobj.predict_cv
    predict_test = tobj.predict_test

    # save results
    with open(f"./0 save data/tobj_{mdl_param['name']}.pkl",'wb') as f:
        pickle.dump(tobj, f)
    shutil.copy(f"./0 save inner data/best_model-{mdl_param['name']}.pth",f"./0 save data/model-{mdl_param['name']}.pth")

    variables = ["fuco","perid","hex_fuco","but_fuco","allo","tchlb","zea","tchla","dvchla"]
    cv_pre = y1_scaler.inverse_transform(predict_cv.copy(),'train_val')
    cv_tar = y1_scaler.inverse_transform(std_y1_train_val.copy(),'train_val')
    test_pre = y1_scaler.inverse_transform(predict_test.copy(),'test')
    test_tar = y1_scaler.inverse_transform(std_y1_test.copy(),'test')
    save_result = {}
    for i in range(predict_cv.shape[1]):
        save_result[variables[i]] = {
            "predict_cv":cv_pre[:,i],
            "target_cv":cv_tar[:,i],
            "predict_test":test_pre[:,i],
            "target_test":test_tar[:,i]
        }
    # scio.savemat("./0 save data/cv_test_result.mat",save_result)

    # print train&val

    a = y1_scaler.inverse_transform(predict_cv.copy(),'train_val')
    b = y1_scaler.inverse_transform(std_y1_train_val.copy(),'train_val')
    for i in range(a.shape[-1]):
        res = data_process.cal_error(a[:,i],b[:,i])
        print("validation",{item:round(res[item],4) for item in res},end="\n")
    a1, b1 = a.flatten(), b.flatten()
    res = data_process.cal_error(a1,b1)
    print("all data validation",{item:round(res[item],4) for item in res},end="\n")


    # print test
    a = y1_scaler.inverse_transform(predict_test.copy(),'test')
    b = y1_scaler.inverse_transform(std_y1_test.copy(),'test')
    for i in range(a.shape[-1]):
        res = data_process.cal_error(a[:,i],b[:,i])
        print("testing",{item:round(res[item],4) for item in res},end="\n")
    a1, b1 = a.flatten(), b.flatten()
    res = data_process.cal_error(a1,b1)
    print("all data testing",{item:round(res[item],4) for item in res},end="\n")

    print(mdl_param["name"] + " done!")





