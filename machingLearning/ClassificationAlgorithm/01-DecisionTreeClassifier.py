#参数：
'''
    criterion:                  可以使用"gini"或者"entropy"，前者代表基尼系数，后者代表信息增益。
    splitter:                   特征划分点选择标准， best 或者 random
    max_features:               划分时考虑的最大特征数，
    max_depth:                  决策树最大深度，默认可以不输入
    min_samples_split:          内部节点再划分所需最小样本数
    min_weight_faction_leaf:    叶子节点最小的样本权重和
    max_leaf_nodes:             最大叶子节点数
    class_weight:               类别权重值
    min_impurity_split:         节点划分最小不纯度
    presort:                    数据是否预排序
'''
from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


# 仍然使用自带的iris数据
iris = datasets.load_iris()
X = iris.data[:, [0, 2]]                                        #数据集
y = iris.target                                                 #标签

# 训练模型，限制树的最大深度4
clf = DecisionTreeClassifier(criterion='entropy',max_depth=4)
#拟合模型
clf.fit(X, y)


# 画图
#生成测试数据
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
#利用模型对测试数据进行分类预测
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)                                #绘制等高线
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.show()