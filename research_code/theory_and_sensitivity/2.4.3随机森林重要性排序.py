import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn import preprocessing

def plot_permutation_importance(clf, X, y, ax):
    result = permutation_importance(clf, X, y, n_repeats=10, random_state=42, n_jobs=2)
    perm_sorted_idx = result.importances_mean.argsort()

    ax.boxplot(
        result.importances[perm_sorted_idx].T,
        vert=False,
        labels=X.columns[perm_sorted_idx],
    )
    ax.axvline(x=0, color="k", linestyle="--")
    return ax
# 读取数据
x_data = np.loadtxt('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/comsol模型/2.4.1sobol/origin_sobol_sample.txt')
X = x_data[:,]
print(X.shape)
y = np.loadtxt('E:\毕设：空间盘绕/0.chapter_data_save/2chaper/comsol模型/2.4.1sobol/2.4.1相对带宽.txt')
print(y.shape)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# 训练模型
clf = RandomForestRegressor(n_estimators=100, random_state=30)
clf.fit(X_train, y_train)
# 预测结果
y_pred = clf.predict(X_test)
# 计算MSE和R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# 输出模型评估结果和目标方程
print('MSE:', mse)
print('R-squared:', r2)
feature_names = ['W','H','w1','w2','w3','w4','w5']
print(feature_names)
print('Gini importance',clf.feature_importances_)

# 绘制特征重要性条形图
feature_importance = clf.feature_importances_
sorted_idx = feature_importance.argsort()#"Gini importance"
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.subplot(1,2,1)
plt.barh(range(len(feature_importance)), feature_importance[sorted_idx])
plt.yticks(range(len(feature_importance)), [feature_names[i] for i in sorted_idx], fontsize=5)
plt.xlabel('特征重要性')
plt.ylabel('特征名称')
plt.title('随机森林回归Gini importance')
#plt.savefig('随机森林回归特征重要性', dpi=300)


#"Decrease in accuracy score"
result = permutation_importance(clf, X, y, n_repeats=10, random_state=42)
perm_sorted_idx = result.importances_mean.argsort()
result = result.importances_mean
print('误差重要性：',result)
plt.subplot(1,2,2)
plt.barh(range(len(feature_importance)), result[perm_sorted_idx])
plt.yticks(range(len(feature_importance)), [feature_names[i] for i in sorted_idx], fontsize=5)

plt.show()
