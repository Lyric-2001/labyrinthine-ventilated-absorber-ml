import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.svm import LinearSVR, SVR
from sklearn.metrics import r2_score
import time
# import xgboost as xgb
# from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.metrics import mean_squared_error
from pylab import *
#支持中文
mpl.rcParams['font.sans-serif'] = ['SimHei']

# fig,axes = plt.subplots(1,1,figsize=(4,4),dpi=100,facecolor="w")
# fig.subplots_adjust(left=0.2,bottom=0.2)
# 自定义训练损失
def custom__train(y_true, y_pred):
    residual = (y_true - y_pred).astype("float")
    a=(1+y_true)
    grad = -2*residual*a
    hess = 2*a
    return grad, hess

# 自定义验证损失
def custom_valid(y_true, y_pred):
    residual = (y_true - y_pred).astype("float")
    loss = (residual / y_true) ** 2
    return "custom_asymmetric_eval", np.mean(loss), False






def picture(y,y_pre):
    plt.figure(figsize=(12, 4))
    plt.plot(range(480,1622,2),y,  label="true")
    plt.plot(range(480,1622,2),y_pre, '--', label="predict")
    plt.title("absorb")
    plt.legend()
    plt.show()
def min_max(parameters,test_parameters):
    min_value = parameters.min(axis=0)
    ptp_value = parameters.ptp(axis=0)
    X = (parameters-min_value)/ptp_value
    test_x = (test_parameters - min_value) /ptp_value
    return X,test_x
def Model_tra(x_train, y_train, x_test, y_test, model):
    start = time.time()
    x_train_scaler,x_test_scaler = min_max(x_train, x_test)
    y_train_scaler, y_test_scaler = min_max(y_train, y_test)
    # 采用模型拟合
    model.fit(x_train_scaler, y_train_scaler.ravel(),eval_set=[(x_test_scaler , y_test_scaler)],
              eval_metric=custom_valid)
    #joblib.dump(model, "./lgbm_model.pkl")
    y_pre = model.predict(x_test_scaler).reshape(-1,1)
    pred = np.array(y_pre).squeeze() * y_train.ptp(axis=0) + y_train.min(axis=0)
    print(dict(zip(model.feature_name_,model.feature_importances_)))
    score = r2_score(y_test,pred )
    mse = mean_squared_error(y_test,pred)**0.5
    print('mse:',mse)
    end = time.time()
    tim = end - start

    y_pred = model.predict(x_test_scaler[0:571]).reshape(-1, 1)
    pred = np.array(y_pred).squeeze() * y_train.ptp(axis=0) + y_train.min(axis=0)
    picture(y_test[0:571],y_pred)
    # 保存


    return score, tim

def peakHZmodel():
    data = np.loadtxt('./train_ParaPeakHz.txt ')
    # np.random.seed(12)
    # np.random.shuffle(data)
    print('train_data.shape', data.shape) 
    x_train = data[:, 0:5]
    y_train = np.around(data[:, 5:6], )

    test_data = np.loadtxt('./test_ParaPeakHz.txt')
    print('test_data.shape', test_data.shape)

    x_test = test_data[:, 0:5]
    y_test = np.around(test_data[:, 5:6], )

    print('x_train.shape,x_test.shape:', x_train.shape, x_test.shape)


    model1 = lgb.LGBMRegressor(n_estimators=500)
    score1, tim1 = Model_tra(x_train, y_train, x_test, y_test, model1)
    print("lightgbm模型的拟合能力：", score1)
    print(f"模型运行时间：%.4f s" % tim1)

    # model4 = KNeighborsRegressor()
    # score4, tim4 = Model_tra(x_train, y_train, x_test, y_test, model4)
    # print("KNN模型的拟合能力：", score4)
    # print(f"模型运行时间：%.4f s" % tim4)#0.994
    #
    # model5 = RandomForestRegressor(n_estimators=30)
    # score5, tim5 = Model_tra(x_train, y_train, x_test, y_test, model5)
    # print("随机森林模型的拟合能力：", score5)
    # print(f"模型运行时间：%.4f s" % tim5)  # 0.991
def Absorbmodel():
    data = np.loadtxt('E:/毕设：空间盘绕/0.chapter_data_save/3chapter/train_ParaHzAbsorb.txt ')
    np.random.seed(5)
    np.random.shuffle(data)
    print('data.shape', data.shape)
    x_train = data[:, 0:6]
    y_train = data[:, 6:7]

    test_data = np.loadtxt('E:/毕设：空间盘绕/0.chapter_data_save/3chapter/test_ParaHzAbsorb.txt')
    print('test_data.shape', test_data.shape)
    x_test = test_data[:, 0:6]
    y_test = test_data[:, 6:7]

    print('x_train.shape,x_test.shape:', x_train.shape, x_test.shape)


    model1 = lgb.LGBMRegressor(n_estimators=1000,objective=custom__train)#,num_leaves=64,max_depth=7
    score1, tim1 = Model_tra(x_train, y_train, x_test, y_test, model1)
    print("lightgbm模型的拟合能力：", score1)
    print(f"模型运行时间：%.4f s" % tim1)


def pic():
    data = np.loadtxt('E:/毕设：空间盘绕/0.chapter_data_save/3chapter/train_ParaHzAbsorb.txt ')
    np.random.seed(5)
    np.random.shuffle(data)
    print('data.shape', data.shape)
    x_train = data[:, 0:6]
    y_train = data[:, 6:7]

    test_data = np.loadtxt('E:/毕设：空间盘绕/0.chapter_data_save/3chapter/test_ParaHzAbsorb.txt')
    print('test_data.shape', test_data.shape)
    x_test = test_data[:, 0:6]
    y_test = test_data[:, 6:7]

    print('x_train.shape,x_test.shape:', x_train.shape, x_test.shape)
    x_train_scaler, x_test_scaler = min_max(x_train, x_test)
    y_train_scaler, y_test_scaler = min_max(y_train, y_test)
    model = joblib.load("./lgbm_model.pkl")

    y_pred = model.predict(x_test_scaler).reshape(-1, 1)
    pred = np.array(y_pred).squeeze() * y_train.ptp(axis=0) + y_train.min(axis=0)

    y_test = y_test.reshape(150,-1,1)
    y_pred = y_pred.reshape(150, -1, 1)
    #n=10
    #picture(y_test[n], y_pred[n])
    plt.figure(figsize=(12, 5))
    # 标签字体设置
    plt.subplot(1, 2, 1)
    plt.plot(range(700, 1302, 2), y_test[0,110:411], label="true",linewidth=2)
    plt.plot(range(700, 1302, 2),y_pred[0,110:411], label="predict",linewidth=2)
    plt.title("test_0", fontdict={'fontsize': 16})
    plt.xlabel('频率(Hz)', fontdict={'fontsize': 16})
    plt.ylabel('吸声系数', fontdict={'fontsize': 16})
    plt.rcParams.update({'font.size': 16})
    plt.yticks(size=14)
    plt.xticks(size=14)
    plt.rcParams.update({'font.size': 14})
    plt.legend(loc='upper right')


    plt.subplot(1, 2, 2)
    plt.plot(range(900, 1502, 2), y_test[10,210:511], label="true",linewidth=2)
    plt.plot(range(900, 1502, 2), y_pred[10,210:511], label="predict",linewidth=2)
    plt.title("test_1", fontdict={'fontsize': 16})
    plt.xlabel('频率(Hz)', fontdict={'fontsize': 16})
    plt.ylabel('吸声系数', fontdict={'fontsize': 16})
    plt.yticks(size=14)
    plt.xticks(size=14)
    plt.rcParams.update({'font.size': 14})
    plt.legend(loc='upper right')
    plt.show()



if __name__ == "__main__":
    # peakhz = 0
    # if peakhz == 1:
    #     peakHZmodel()
    # else:
    #     Absorbmodel()

    # 加载
    #
    pic()




