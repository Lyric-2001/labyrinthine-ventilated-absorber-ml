import logging
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
#全文过滤警告
import warnings
from sklearn.metrics import r2_score
import os
from torch.utils.data import Dataset,DataLoader
gpu = 0
use_cuda = gpu >= 0 and torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
else:
    device = torch.device("cpu")
logging.info("Use cuda: %s, gpu id: %d.", use_cuda, gpu)

class MyDataset(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index],self.y[index]

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.x)
class CNN_LSTM(nn.Module):
    def __init__(self,n_filter,conv_input_channel,conv_output_channel, hidden_size, num_layers, output_size,dropout,training):
        super(CNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.conv_output_channel = conv_output_channel
        self.dropout = nn.Dropout(dropout)
        #卷积核
        self.filter_sizes = [(1+x) for x in range(1,n_filter+1)]  # n-gram window
        # 3 个卷积层，卷积核大小分别为 [2,1], [3,1], [4,1]
        #self.convs = nn.ModuleList([nn.Conv1d(conv_input_channel, conv_output_channel, kernel_size = filter_size)
               #                     for filter_size in self.filter_sizes])


        #lstm+linear
        #self.lstm = nn.LSTM(self.conv_output_channel*len(self.filter_sizes),
                         #   hidden_size, num_layers, batch_first = True)
        self.fc = nn.Sequential(nn.Linear(in_features=9,out_features=128),nn.ReLU(),
                                nn.Linear(in_features=128,out_features=512),nn.ReLU(),
                                nn.Linear(in_features=512,out_features=128),nn.ReLU(),
                                nn.Linear(in_features=128,out_features=32),nn.ReLU(),
                                nn.Linear(32,output_size))
        self.training = training
    def forward(self, x):
        # pooled_outputs = []
        # #print('x.shape',x.shape)
        # # 5.相加后的词向量 通过 3 个卷积核做 3 次卷积核池化 并拼接
        # for i in range(len(self.filter_sizes)):
        #     filter_height = x.shape[-1] - self.filter_sizes[i] + 1
        #     # conv：sentence_num * out_channel * filter_height * 1
        #     conv = self.convs[i](x)
        #     #print("conv.shape:", conv.shape)
        #     r= nn.ReLU()
        #     hidden = r(conv)
        #     #print("hidden.shape:", hidden.shape)
        #     # 定义池化层
        #     mp = nn.MaxPool1d(kernel_size=filter_height)  # (filter_height, filter_width)
        #     # pooled：sentence_num * out_channel * 1 * 1 -> sen_num * out_channel
        #     # 也可以通过 squeeze 来删除无用的维度
        #
        #     pooled = mp(hidden)
        #     #print("pooled.shape:",pooled.shape)
        #     #.reshape(sen_num,-1)
        #
        #     pooled_outputs.append(pooled)
        # # 拼接 3 个池化后的向量
        # # reps: sen_num * (3*out_channel)
        # reps = torch.cat(pooled_outputs, dim=1)
        # #print('reps.shape',reps.shape)
        #
        # if self.training:
        #     reps = self.dropout(reps)
        #
        # out, _= self.lstm(reps.permute(0,2,1))
        # # out, _ = self.lstm(reps.permute(2, 0, 1))
        #print('x.shape',x.shape)
        out = self.fc(x)
        #print('out.shape',out.shape)
        return out


class EarlyStopping():
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        print("val_loss={}".format(val_loss), end='\t')
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + 'Lstm.pth')
        self.val_loss_min = val_loss


def loss_picture(train_loss, train_epochs_loss, valid_epochs_loss):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss[:])
    plt.title("train_loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_epochs_loss[1:], label="train_loss")
    plt.plot(valid_epochs_loss[1:], label="valid_loss")
    plt.title("epochs_loss")
    plt.legend()
    plt.show()


def train2(train_dataloader, test_dataloader, epochs,
           optimizer, model, criterion, early_stopping, batch_size):
    train_epochs_loss = []
    valid_epochs_loss = []
    train_loss = []
    valid_loss = []
    for epoch in range(1, epochs + 1):
        logging.info("--" * 10)
        model.train()
        train_epoch_loss = []
        for idx, (x, y) in enumerate(train_dataloader, 1):
            data_x = x.to(torch.float32).to(device)
            data_y = y.to(torch.float32).to(device)
            # print('data_y.shape,data_x.shape',data_y.shape,data_x.shape)
            optimizer.zero_grad()
            outputs = model(data_x)

            loss = criterion(data_y, outputs)
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
            train_loss.append(loss.item())
            if idx % (100) == 0:  # (len(train_data) // 10)
                print("epoch={},{}of train, loss={}".format(
                    epoch, idx * batch_size, loss.item()))
        train_epochs_loss.append(np.average(train_epoch_loss))

        # =====================valid============================
        model.eval()
        valid_epoch_loss = []
        with torch.no_grad():
            for idx, (x, y) in enumerate(test_dataloader, 1):
                data_x = x.to(torch.float32).to(device)
                data_y = y.to(torch.float32).to(device)
                outputs = model(data_x)
                loss = criterion(data_y, outputs)
                valid_epoch_loss.append(loss.item())
                valid_loss.append(loss.item())
        valid_epochs_loss.append(np.average(valid_epoch_loss))

        # ==================early stopping======================
        print(f"epoch={epoch}", end='\t')
        early_stopping(valid_epochs_loss[-1], model=model, path=r'./')
        if early_stopping.early_stop:
            print("Early stopping")
            break
        # ====================adjust lr========================
        lr_adjust = {5: 0.005, 10: 0.001, 20: 5e-4, 30: 1e-4}
        if epoch in lr_adjust:
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('Updating learning rate to {}'.format(lr))

        # ====================adjust lr========================
    loss_picture(train_loss, train_epochs_loss, valid_epochs_loss)


def min_max(parameters, test_parameters):
    min_value = parameters.min(axis=0)
    ptp_value = parameters.ptp(axis=0)
    X = (parameters - min_value) / ptp_value
    test_x = (test_parameters - min_value) / ptp_value
    return X, test_x


def test():
    data = np.loadtxt('/kaggle/input/xishu-300/_300.txt', delimiter='\t')
    x = data[:, 0:9]
    y = data[:, 9:11]
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    x_train_scaler, x_test_scaler = min_max(x_train, x_test)
    y_train_scaler, y_test_scaler = min_max(y_train, y_test)

    batch_size = 64
    train_dataset = MyDataset(x_train_scaler, y_train_scaler)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    test_dataset = MyDataset(x_test_scaler, y_test_scaler)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # =============================================================================================
    # =========================cnn_lstm parameter========================================
    filter_size, conv_input_channel, conv_output_channel = 4, 1, 64  # cnn
    hidden_size, num_layers = 512, 2  # LSTM
    output_size = 2  # Fnn
    dropout = 0.15
    training = True
    model = CNN_LSTM(filter_size, conv_input_channel, conv_output_channel,
                     hidden_size, num_layers, output_size, dropout, training)
    model.to(device)
    # =========================init========================================
    # 设置损失函数,这里使用的是均方误差损失
    criterion = nn.MSELoss()
    # 设置优化函数和学习率lr
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # 设置训练周期
    epochs = 1000
    patience = 6  # 早停迭代次数

    early_stopping = EarlyStopping(patience=patience, verbose=True)
    # ================================================================
    train2(train_dataloader, test_dataloader, epochs,
           optimizer, model, criterion, early_stopping, batch_size)
    # =============================predict=========================================
    model.load_state_dict(torch.load('./Lstm.pth'))
    # pre = model_predict(model, x_test, y_test, scaler_x, scaler_y)

    test_dataset = MyDataset(x_test_scaler, y_test_scaler)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    y_pred = []
    with torch.no_grad():
        for idx, (x, y) in enumerate(test_dataloader, 1):
            data_x = x.to(torch.float32).to(device)
            y_pred.append(model(data_x).reshape(-1, ).cpu())
    logging.info('===========================================')
    pred = np.array(y_pred) * y_train.ptp(axis=0) + y_train.min(axis=0)
    print('pred.shape:', pred.shape)
    print('r2_score=%.4f' % (r2_score(y_test, pred)))
def load_model_predict():
    data = np.loadtxt('/kaggle/input/xishu-300/_300.txt', delimiter='\t')
    x = data[:, 0:9]
    y = data[:, 9:11]
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    x_train_scaler, x_test_scaler = min_max(x_train, x_test)
    y_train_scaler, y_test_scaler = min_max(y_train, y_test)

    # =============================================================================================
    # =========================cnn_lstm parameter========================================
    filter_size, conv_input_channel, conv_output_channel = 4, 1, 64  # cnn
    hidden_size, num_layers = 512, 2  # LSTM
    output_size = 2  # Fnn
    dropout = 0.15
    training = True
    model = CNN_LSTM(filter_size, conv_input_channel, conv_output_channel,
                     hidden_size, num_layers, output_size, dropout, training)
    model.to(device)
    # =========================init========================================
    # 设置损失函数,这里使用的是均方误差损失

    # =============================predict=========================================
    model.load_state_dict(torch.load('./Lstm.pth'))
    # pre = model_predict(model, x_test, y_test, scaler_x, scaler_y)

    test_dataset = MyDataset(x_test_scaler, y_test_scaler)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    y_pred = []
    with torch.no_grad():
        for idx, (x, y) in enumerate(test_dataloader, 1):
            data_x = x.to(torch.float32).to(device)
            y_pred.append(model(data_x).reshape(-1, ).cpu())
    logging.info('===========================================')
    pred = np.array(y_pred) * y_train.ptp(axis=0) + y_train.min(axis=0)
    print('pred.shape:', pred.shape)
    print('r2_score=%.4f' % (r2_score(y_test, pred)))
    y_test[0:2], pred[0:2]

if __name__ == "__main__":
    test()