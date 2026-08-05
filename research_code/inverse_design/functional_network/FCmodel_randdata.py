from FCmodel import FCmodel,min_max,DataLoader,MyDataset
#[ 5.5,831.96127696 ,2.97 ,0.856,20.119,6.9,16.76]
import torch
import numpy as np
def FcPredict(x_test_scaler,y_test_scaler,trainptp,trainmin):
    #=====================加载模型==========================
    output_size = len(y_test_scaler[0]) # Fnn
    training = False
    model = FCmodel(output_size, training)#(output_size,training)

    model.load_state_dict(torch.load('./4.2model1_0.9997_r0_6e-6.pth'))
    model.to(device)
    #==================测试集预测=======================
    test_dataset = MyDataset(x_test_scaler, y_test_scaler)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    y_pred = []
    with torch.no_grad():
        for idx, (x, y) in enumerate(test_dataloader, 1):
            data_x = x.to(torch.float32).to(device)
            y_pred.append(model(data_x).reshape(1, -1).cpu())
    print('=================预测结束-评价指标====================')
    pred = np.array(y_pred).squeeze() * trainptp + trainmin
    print('pred.shape:', pred.shape)
    return pred
if __name__=="__main__":
    gpu = 0
    use_cuda = gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(gpu)
        device = torch.device("cuda", gpu)
    else:
        device = torch.device("cpu")
    print("Use cuda: %s, gpu id: %d.", use_cuda, gpu)
    # ============================读取数据===================
    data = np.loadtxt('./latin_train_absor_para.txt')
    print('data.shape', data.shape)
    x_train = data[:, 0:-1]
    y_train = data[:, -1:]
    #========================任意数据=======
    HZ= np.array(range(480,1622,2))
    x_test = np.array([812,832,846,0.856,20.1,6.9,16.8])
    new_para = x_test.reshape(1, -1).repeat(571, axis=0)
    x_test = np.concatenate((new_para, HZ.reshape(-1, 1)), axis=1)
    y_test = np.random.rand(571,1)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    x_train_scaler, x_test_scaler = min_max(x_train, x_test)
    y_train_scaler, y_test_scaler = min_max(y_train, y_test)
    trainptp = y_train.ptp(axis=0)
    trainmin = y_train.min(axis=0)
    # =========================预测==============================预测
    pred = FcPredict(x_test_scaler, y_test_scaler, trainptp, trainmin)
    np.savetxt('./y_pre_nonel_r_0.991.txt', pred.reshape(-1, 1))