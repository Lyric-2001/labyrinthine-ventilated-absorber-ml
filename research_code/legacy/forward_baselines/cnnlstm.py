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


os.environ['KMP_DUPLICATE_LIB_OK']='True'

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s-%(funcName)s')

# set seed
seed = 666
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)

# set cuda
gpu = 0
use_cuda = gpu >= 0 and torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
else:
    device = torch.device("cpu")
logging.info("Use cuda: %s, gpu id: %d.", use_cuda, gpu)

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
        self.convs = nn.ModuleList([nn.Conv1d(conv_input_channel, conv_output_channel, kernel_size = filter_size)
                                    for filter_size in self.filter_sizes])


        #lstm+linear
        self.lstm = nn.LSTM(self.conv_output_channel*len(self.filter_sizes),
                            hidden_size, num_layers, batch_first = True)
        self.fc = nn.Sequential(nn.Linear(in_features=self.hidden_size,out_features=128),nn.Tanh(),
                                nn.Linear(128,output_size))
        self.training = training
    def forward(self, x):
        pooled_outputs = []
        #print('x.shape',x.shape)
        # 5.相加后的词向量 通过 3 个卷积核做 3 次卷积核池化 并拼接
        for i in range(len(self.filter_sizes)):
            filter_height = x.shape[-1] - self.filter_sizes[i] + 1
            # conv：sentence_num * out_channel * filter_height * 1
            conv = self.convs[i](x)
            #print("conv.shape:", conv.shape)
            r= nn.ReLU()
            hidden = r(conv)
            #print("hidden.shape:", hidden.shape)
            # 定义池化层
            mp = nn.MaxPool1d(kernel_size=filter_height)  # (filter_height, filter_width)
            # pooled：sentence_num * out_channel * 1 * 1 -> sen_num * out_channel
            # 也可以通过 squeeze 来删除无用的维度

            pooled = mp(hidden)
            #print("pooled.shape:",pooled.shape)
            #.reshape(sen_num,-1)

            pooled_outputs.append(pooled)
        # 拼接 3 个池化后的向量
        # reps: sen_num * (3*out_channel)
        reps = torch.cat(pooled_outputs, dim=1)
        #print('reps.shape',reps.shape)

        if self.training:
            reps = self.dropout(reps)

        out, _= self.lstm(reps.permute(0,2,1))
        # out, _ = self.lstm(reps.permute(2, 0, 1))
        #print('out.shape',out.shape)
        out = self.fc(out.squeeze())

        return out
def batch_slice(data, batch_size):
    # batch_num:划分多少个批次
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        # 如果 i < batch_num - 1，那么大小为 batch_size，否则就是最后一批数据
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        # 划分为batch_num个批次，每个批次batch_size个新闻
        docs = [data[i * batch_size + b] for b in range(cur_batch_size)]
        yield torch.tensor(docs)
def data_iter(x,y, batch_size, shuffle=True):
    data = np.concatenate((x,y),axis=1)
    batched_data = []
    if shuffle:
        # 这里是打乱所有数据
        np.random.shuffle(data)
    # 调用batch_slice函数，把 batch 的数据放进一个 list
    batched_data.extend(list(batch_slice(data, batch_size)))
    if shuffle:
        # 打乱 多个 batch
        np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch
class EarlyStopping():
    def __init__(self,patience=7,verbose=False,delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
    def __call__(self,val_loss,model,path):
        print("val_loss={}".format(val_loss))
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss,model,path)
        elif score < self.best_score+self.delta:
            self.counter+=1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter>=self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss,model,path)
            self.counter = 0
    def save_checkpoint(self,val_loss,model,path):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'CnnLstm.pth')
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

def train2(x_train_scaler,y_train_scaler,x_test_scaler,y_test_scaler
           ,epochs,optimizer,model,criterion,early_stopping,batch_size):
    train_epochs_loss = []
    valid_epochs_loss = []
    train_loss = []
    valid_loss = []
    for epoch in range(1,epochs+1):
        logging.info("--"*10)
        model.train()
        train_epoch_loss = []
        for idx,batch in enumerate(data_iter(x_train_scaler,y_train_scaler, batch_size = batch_size, shuffle=True),1):
            data_x = batch[:,0:-1].unsqueeze(1).to(torch.float32)
            data_y = batch[:,-1].to(torch.float32)
            #print('data_y.shape,data_x.shape',data_y.shape,data_x.shape)
            optimizer.zero_grad()
            outputs = model(data_x)

            loss = criterion(data_y, outputs)
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
            train_loss.append(loss.item())
            if idx % (100) == 0:# (len(train_data) // 10)
                print("epoch={},{}/{}of train, loss={}".format(
                    epoch, idx*batch_size, len(x_train_scaler), loss.item()))
        train_epochs_loss.append(np.average(train_epoch_loss))

        # =====================valid============================
        model.eval()
        valid_epoch_loss = []
        with torch.no_grad():
            for idx,batch in enumerate(data_iter(x_test_scaler,y_test_scaler, batch_size = batch_size,
                                                 shuffle=True),1):
                data_x = batch[:,0:-1].unsqueeze(1).to(torch.float32)
                data_y = batch[:,-1].to(torch.float32)
                outputs = model(data_x)
                loss = criterion(data_y, outputs)
                valid_epoch_loss.append(loss.item())
                valid_loss.append(loss.item())
        valid_epochs_loss.append(np.average(valid_epoch_loss))

        # ==================early stopping======================
        early_stopping(valid_epochs_loss[-1], model=model, path=r'./')
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # ====================adjust lr========================
        lr_adjust = {3: 3e-4, 6: 2e-4, 10: 1e-4, 15: 5e-5}
        if epoch in lr_adjust:
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('Updating learning rate to {}'.format(lr))
    loss_picture(train_loss, train_epochs_loss, valid_epochs_loss)

def model_predict(model, x_test,y_test,scaler_x,scaler_y):
    # 此处可定义一个预测集的Dataloader。也可以直接将你的预测数据reshape,添加batch_size=1
    model.eval()
    y_pred = []
    x_test_scaler = scaler_x.transform(x_test)
    with torch.no_grad():
        for idx, batch in enumerate(data_iter(x_test_scaler,y_test, batch_size=1, shuffle=False), 1):
            data_x = batch[:, 0:-1].unsqueeze(1).to(torch.float32)
            y_pred.append(model(data_x).reshape(-1,))


    logging.info('===========================================')
    print('r2_score=%f'%(r2_score(y_test, scaler_y.inverse_transform(y_pred))))

def test():
    data = np.loadtxt('F:/深度学习/608项目程序/稀疏采样_300.txt',delimiter='\t')
    x = data[:,0:9]
    y =data[:,9:10]

    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.2,shuffle=True)
    print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
    scaler_x = MinMaxScaler().fit(x_train)
    x_train_scaler = scaler_x.transform(x_train)
    x_test_scaler = scaler_x.transform(x_test)
    scaler_y = MinMaxScaler().fit(y_train)
    y_train_scaler = scaler_y.transform(y_train)
    y_test_scaler = scaler_y.transform(y_test)

    # =========================cnn_lstm parameter========================================
    filter_size, conv_input_channel, conv_output_channel = 4, 1, 64  # cnn
    hidden_size, num_layers = 512, 2  # LSTM
    output_size = 1  # Fnn
    dropout = 0.15
    training = True
    model = CNN_LSTM(filter_size, conv_input_channel, conv_output_channel,
                     hidden_size, num_layers, output_size, dropout, training)
    # =========================init========================================
    # 设置损失函数,这里使用的是均方误差损失
    criterion = nn.MSELoss()
    # 设置优化函数和学习率lr
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # 设置训练周期
    epochs = 5
    patience = 5  # 早停迭代次数
    batch_size = 16
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    # ================================================================

    #================================================================
    train2(x_train_scaler, y_train_scaler, x_test_scaler, y_test_scaler,
           epochs, optimizer, model, criterion, early_stopping, batch_size)
    # =============================predict=========================================
    model.load_state_dict(torch.load('./CnnLstm.pth'))
    #pre = model_predict(model, x_test, y_test, scaler_x, scaler_y)

    y_pred = []
    with torch.no_grad():
        for idx, batch in enumerate(data_iter(x_test_scaler,y_test_scaler, batch_size=1, shuffle=False), 1):
            data_x = batch[:, 0:-1].unsqueeze(1).to(torch.float32)
            y_pred.append(model(data_x).reshape(-1,))
    logging.info('===========================================')
    print('r2_score=%f' % (r2_score(y_test, scaler_y.inverse_transform(y_pred))))


def absorb_text():
    data = np.loadtxt('E:/毕设：空间盘绕/0.chapter_data_save/3chapter/train_ParaHzAbsorb.txt ')
    print('data.shape', data.shape)
    x_train = data[:, 0:6]
    y_train = data[:, 6:7]

    test_data = np.loadtxt('E:/毕设：空间盘绕/0.chapter_data_save/3chapter/test_ParaHzAbsorb.txt')
    print('test_data.shape', test_data.shape)
    x_test = test_data[:, 0:6]
    y_test = test_data[:, 6:7]

    print('x_train.shape,x_test.shape:', x_train.shape, x_test.shape)
    # ===========================MinMaxScaler============================
    scaler_x = MinMaxScaler().fit(x_train)
    x_train_scaler = scaler_x.transform(x_train)
    x_test_scaler = scaler_x.transform(x_test)
    scaler_y = MinMaxScaler().fit(y_train)
    y_train_scaler = scaler_y.transform(y_train)
    y_test_scaler = scaler_y.transform(y_test)
    # =========================cnn_lstm parameter========================================
    filter_size, conv_input_channel, conv_output_channel = 4, 1, 12  # cnn
    hidden_size, num_layers = 150, 2  # LSTM
    output_size = 1  # Fnn
    dropout = 0.15
    training = True
    model = CNN_LSTM(filter_size, conv_input_channel, conv_output_channel,
                     hidden_size, num_layers, output_size, dropout, training)
    # =========================init========================================
    # 设置损失函数,这里使用的是均方误差损失
    criterion = nn.MSELoss()
    # 设置优化函数和学习率lr
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # 设置训练周期
    epochs = 10
    patience = 5  # 早停迭代次数
    batch_size = 256
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    # ==========================train=============================================
    train2(x_train_scaler, y_train_scaler, x_test_scaler, y_test_scaler,
           epochs, optimizer, model, criterion, early_stopping, batch_size)
    # =============================predict=========================================
    model.load_state_dict(torch.load('./CnnLstm.pth'))
    pre = model_predict(model, x_test, y_test, scaler_x, scaler_y)


if __name__ == "__main__":
    test()

