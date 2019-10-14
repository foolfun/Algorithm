#coding:utf-8

import operator
from sklearn.datasets import load_iris
import numpy as np

#通过KNN进行分类
def classify(inx,dataset,target,k):
    dataSetSize = dataset.shape[0]
    #计算欧式距离,对公式的翻译
    diff = np.tile(inx,(dataSetSize,1)) - dataset
    sqdiff = diff ** 2
    squareDist =sqdiff.sum(axis = 1)#行向量分别相加，从而得到新的一个行向量
    dist = squareDist ** 0.5
    #对距离进行排序
    sortedDistIndex = np.argsort(dist)##argsort()根据元素的值从大到小对元素进行排序，返回下标
    classCount={}
    for i in range(k):
        col = sortedDistIndex[i]
        voteLabel = target[col]
        #对选取的K个样本所属的类别个数进行统计
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    #选取出现的类别次数最多的类别
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]#取投票结果最大的label


data=load_iris()
dataset = data.data
labels = data.feature_names
target = data.target
# 使数据集乱序
num_example = dataset.shape[0]
array = np.arange(num_example)
np.random.shuffle(array)
dataset = dataset[array]
target = target[array]
# 训练集: 验证集 = 9: 3
sample = np.int(num_example * 0.9)
x_train = dataset[: sample]
y_train = target[: sample]
x_test = dataset[sample:]
y_test = target[sample:]
res =[]
for k in range(10):
    pri = []
    tmp = 0
    prob = 0.1
    for j in range(10):
        i=0;cnt=0
        for line in x_test:
            pre = classify(line,x_train[tmp:np.int(sample*prob)],y_train[tmp:np.int(sample*prob)],3)
        #    print(line,"的预测结果为：",pre,'实际结果为:',y_test[i])
            if pre == y_test[i]:
                cnt +=1
            i +=1
        tmp=np.int(sample*prob)+1
        prob += 0.1
#        print('第',j+1,'次预测准确率为:',cnt/len(y_test))
        pri.append(cnt/len(y_test))
    res.append(sum(pri)/len(pri))
print('KNN，10次十折交叉验证结果:',sum(res)/len(res))
