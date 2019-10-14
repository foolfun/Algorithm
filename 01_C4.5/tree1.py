# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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

#ID3
def chooseBsetFeatureToSplit(dataset):
    '''
     得到最佳的分裂点的列号
    '''
    numfeatures = len(dataset[0])-1
    baseEnt = calEntropy(dataset)
    bestInfoGain = 0.0;bestFeat = -1
    for i in range(numfeatures):
        featlist = [line[i] for line in dataset]
        featset = set(featlist)
        newEnt = 0.0
        for value in featset:
            subDataset = splitData(dataset,i,value)
            prob = len(subDataset)/float(len(dataset))
            newEnt += prob * calEntropy(subDataset)
        infoGain = baseEnt - newEnt
        if infoGain>bestInfoGain:
            bestInfoGain = infoGain
            bestFeat = i
    return bestFeat
    
#C4.5
def C45chooseBsetFeatureToSplit(dataset):
    '''
     得到最佳的分裂点的列号
    '''
    numfeatures = len(dataset[0])-1
    baseEnt = calEntropy(dataset)
    bestGainRatio = 0.0
    bestFeat = -1
    split = 0.0
    for i in range(numfeatures):
        featlist = [line[i] for line in dataset]
        featset = set(featlist)
        newEnt = 0.0
        for value in featset:
            subDataset = splitData(dataset,i,value)
            prob = len(subDataset)/float(len(dataset))
            newEnt += prob * calEntropy(subDataset)
            split -= prob * log(prob,2)
        infoGain = baseEnt - newEnt
        GainRatio = infoGain/split
        print(u"C4.5中第%d个特征的信息增益率为：%.3f"%(i,GainRatio))
        if GainRatio>bestGainRatio:
            bestGainRatio = GainRatio
            bestFeat = i
    return bestFeat

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

def creatTree(dataset,labels):
    '''
      递归建树，可能有点难理解，需要好好想想，可以举例：
      bestFeat=2;
      bestFeatLabel=‘天气’;
      value=‘晴’;
      un_featValues=[‘晴’，‘阴’，‘雨’]
      myTree[‘天气’][‘晴’]=。。。
    '''
    classlist = [line[-1] for line in dataset]
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]
    if len(dataset[0]) == 1:
        return majorityCnt(classlist)
    bestFeat = chooseBsetFeatureToSplit(dataset)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [line[bestFeat] for line in dataset]
    un_featValues = set(featValues)
    for value in un_featValues:
        subLables = labels[:]
        myTree[bestFeatLabel][value] = creatTree(splitData(dataset,bestFeat,value),subLables)
    return myTree
 
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
    print(u"此时最优索引为："+(bestFeatLabel))
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

def classifytest(inputTree, featLabels, testDataSet):
    """
    输入：决策树，分类标签，测试数据集
    输出：决策结果
    描述：跑决策树
    """
    classLabelAll = []
    for testVec in testDataSet:
        classLabelAll.append(classify(inputTree, featLabels, testVec))
    return classLabelAll

if __name__ == '__main__':
    data=load_iris()
    dataset = data.data
    labels = data.feature_names
    target = data.target
    factor0 = pd.cut(dataset[0],3)
    factor1 = pd.cut(dataset[1],3)
    factor2 = pd.cut(dataset[2],3)
    factor3 = pd.cut(dataset[3],3)
    for line in dataset:
        d0 = line[0]
        d1 = line[1]
        d2 = line[2]
        d3 = line[3]
        if d0<1.833:
            line[0] = 1
        elif d0>3.467:
            line[0] = 2
        else:
            line[0] = 3
        if d1<1.133:
            line[1] = 1
        elif d1>2.067:
            line[1] = 2
        else:
            line[1] = 3
        if d2<1.2:
            line[2] = 1
        elif d2>2.2:
            line[2] = 2
        else:
            line[2] = 3
        if d3<1.167:
            line[3] = 1
        elif d3>2.133:
            line[3] = 2
        else:
            line[3] = 3
        
    # 使数据集乱序
    num_example = dataset.shape[0]
    array = np.arange(num_example)
    np.random.shuffle(array)
    dataset = dataset[array]
    target = target[array]
    # 训练集: 验证集 = 7: 3, 考虑到样本较少，验证集的结果可以反映测试集结果
    sample = np.int(num_example * 0.7)
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

    labels_tmp = labels[:] # 拷贝，createTree会改变labels        
    CARTdesicionTree = CART_createTree(dataset,labels_tmp)
    print('CARTdesicionTree:\n', CARTdesicionTree)
    treePlotter.CART_Tree(CARTdesicionTree)
    print("下面为测试数据集结果：")
    print('CART_TestSet_classifyResult:\n', classifytest(CARTdesicionTree, labels, dataset_test))        