# -*- coding: utf-8 -*-
"""
Created on Thu May  9 19:12:25 2019

@author: zsl
"""

from math import log
import operator
import treePlotter
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

def calEntropy(dataset):
    '''
    计算熵
    '''
    data_dic ={}
    total = len(dataset)
    ent = 0.0
    for line in dataset:
        label = line[-1]
        if label not in data_dic.keys():
            data_dic[label] = 0
        data_dic[label] += 1
    for key in data_dic:
        prob = float(data_dic[key])/total
        ent -= prob*log(prob,2)
    return ent

def splitData(dataset,i,value):
    '''
     以第i列值为value的进行划分，得到划分结果为value的数据集，但是此时不包含第i列
    '''
    redataset =[]
    for line in dataset:
        if line[i] == value:
            reline = line[:i]
            reline.extend(line[i+1:])
            redataset.append(reline)
    return redataset

def majorityCnt(classList):
    '''
      在遇到属性都用完了之后，还存在不一致的情况  投票选出分类结果
    '''
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] =0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#CART算法
def CART_chooseBestFeatureToSplit(dataset):

    numFeatures = len(dataset[0]) - 1
    bestGini = 999999.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataset]
        uniqueVals = set(featList)
        gini = 0.0
        for value in uniqueVals:
            subdataset=splitData(dataset,i,value)
            p=len(subdataset)/float(len(dataset))
            subp = len(splitData(subdataset, -1, '0')) / float(len(subdataset))
        gini += p * (1.0 - pow(subp, 2) - pow(1 - subp, 2))
#        print(u"CART中第%d个特征的基尼值为：%.3f"%(i,gini))
        if (gini < bestGini):
            bestGini = gini
            bestFeature = i
    return bestFeature 

def CART_createTree(dataset,labels):
    classList=[example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList):
        # 类别完全相同，停止划分
        return classList[0]
    if len(dataset[0]) == 1:
        # 遍历完所有特征时返回出现次数最多的
        return majorityCnt(classList)
    bestFeat = CART_chooseBestFeatureToSplit(dataset)
    #print(u"此时最优索引为："+str(bestFeat))
    bestFeatLabel = labels[bestFeat]
#    print(u"此时最优索引为："+(bestFeatLabel))
    CARTTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    # 得到列表包括节点所有的属性值
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        CARTTree[bestFeatLabel][value] = CART_createTree(splitData(dataset, bestFeat, value), subLabels)
    return CARTTree 

def classify(inputTree, featLabels, testVec):
    """
    输入：决策树，分类标签，测试数据
    输出：决策结果
    描述：跑决策树
    """
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    classLabel = '0'
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def classifytest(inputTree, featLabels, testDataSet,target_test):
    """
        测试
    """
    i =0
    cnt = 0
    for testVec in testDataSet:
        pre = classify(inputTree, featLabels, testVec)
        if pre == target_test[i]:
            cnt +=1
    return cnt/len(target_test)

if __name__ == '__main__':
    data=load_iris()
    dataset = data.data
    labels = data.feature_names
    target = data.target
    dataset[:,0]=pd.cut(dataset[:,0],2,labels=[1,2])
    dataset[:,1]=pd.cut(dataset[:,1],2,labels=[1,2])
    dataset[:,2]=pd.cut(dataset[:,2],2,labels=[1,2])
    dataset[:,3]=pd.cut(dataset[:,3],2,labels=[1,2])
    
    for i in range(len(target)):
        if target[i] == 2:
            target[i] = 0

    # 使数据集乱序
    num_example = dataset.shape[0]
    array = np.arange(num_example)
    np.random.shuffle(array)
    dataset = dataset[array]
    target = target[array]
    # 训练集: 验证集 = 9: 1
    sample = np.int(num_example * 0.9)
    x_train = dataset[: sample]
    y_train = target[: sample]
    x_test = dataset[sample:]
    y_test = target[sample:]
    
    dataset = x_train.tolist()
    target = y_train.tolist()
    for i in range(len(dataset)):
        dataset[i].extend([target[i]])
    
    dataset_test = x_test.tolist()
    target_test = y_test.tolist()
    for i in range(len(dataset_test)):
        dataset_test[i].extend([target_test[i]])

    res=[]
    for j in range(10):
        pri = []
        tmp = 0
        prob = 0.1
        for i in range(10):
            labels_tmp = labels[:] # 拷贝，createTree会改变labels  
            CARTdesicionTree = CART_createTree(dataset[tmp:np.int(sample*prob)],labels_tmp[tmp:np.int(sample*prob)])
    #        print('CARTdesicionTree:\n', CARTdesicionTree)
    #        treePlotter.CART_Tree(CARTdesicionTree)
            pre = classifytest(CARTdesicionTree, labels, dataset_test,target_test)
            pri.append(pre)
            tmp=np.int(sample*prob)+1
            prob += 0.1
        res.append(sum(pri)/len(pri))
    print('CART分类树，10次十折交叉验证结果:',sum(pri)/len(pri))  