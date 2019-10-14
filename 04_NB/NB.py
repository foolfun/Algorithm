# -*- coding: utf-8 -*-
"""
Created on Thu May 16 17:26:19 2019

@author: VULCANTSeries
"""
import numpy as np
import pandas as pd

# 读取数据
def loadDataSet():
    data = pd.read_csv(r"C:\Users\VULCANTSeries\Desktop\z\Skin_NonSkin.txt",names=["x1","x2","x3","y0"],delimiter="\t")
    data['x1']=pd.cut(data['x1'],2,labels=[0,1])
    data['x2']=pd.cut(data['x2'],2,labels=[0,1])
    data['x3']=pd.cut(data['x3'],2,labels=[0,1])
    
    data['y']=data['y0']
    data.loc[data['y0']==1,'y']=0
    data.loc[data['y0']==2,'y']=1
    
    postingList = data[['x1','x2','x3']].values
    classVec = data['y'].values
    return postingList, classVec


# 朴素贝叶斯分类器训练函数   从词向量计算概率
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # p0Num = zeros(numWords); p1Num = zeros(numWords)
    # p0Denom = 0.0; p1Denom = 0.0
    p0Num = np.ones(numWords);   # 避免一个概率值为0,最后的乘积也为0
    p1Num = np.ones(numWords);   # 用来统计两类数据中，各词的词频
    p0Denom = 2.0;  # 用于统计0类中的总数
    p1Denom = 2.0  # 用于统计1类中的总数
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
            # p1Vect = p1Num / p1Denom
            # p0Vect = p0Num / p0Denom
    p1Vect = np.log(p1Num / p1Denom)    # 在类1中，每个次的发生概率
    p0Vect = np.log(p0Num / p0Denom)      # 避免下溢出或者浮点数舍入导致的错误   下溢出是由太多很小的数相乘得到的
    return p0Vect, p1Vect, pAbusive

# 朴素贝叶斯分类器
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify*p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify*p0Vec) + np.log(1.0-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def classifytest(testDataSet, p0Vec, p1Vec, pClass1,target_test):
    """
        测试
    """
    i =0
    cnt = 0
    for testVec in testDataSet:
        pre = classifyNB(testVec, p0Vec, p1Vec, pClass1)
        if pre == target_test[i]:
            cnt +=1
        i += 1
    return cnt/len(target_test)

def testingNB():
    dataset, target = loadDataSet()
    num_example = dataset.shape[0]
    array = np.arange(num_example)
    np.random.shuffle(array)
    dataset = dataset[array]
    target = target[array]
    sample = np.int(num_example * 0.9)
    x_train = dataset[: sample].tolist()
    y_train = target[: sample].tolist()
    x_test = dataset[sample:].tolist()
    y_test = target[sample:].tolist()
#    p0V, p1V, pAb = trainNB0(dataset.tolist(), target.tolist())
#    testEntry = [1,1,0]
#    print (testEntry, 'classified as: ', classifyNB(testEntry, p0V, p1V, pAb))
#    testEntry = [0,1,1]
#    print (testEntry, 'classified as: ', classifyNB(testEntry, p0V, p1V, pAb))
    res=[]
    for j in range(10):
        pri = []
        tmp = 0
        prob = 0.1
        for i in range(10):
            p0V, p1V, pAb = trainNB0(x_train[tmp:np.int(sample*prob)],y_train)
            pre = classifytest(x_test, p0V, p1V, pAb,y_test)
            pri.append(pre)
            tmp=np.int(sample*prob)+1
            prob += 0.1
        res.append(sum(pri)/len(pri))
        print(sum(pri)/len(pri))
    print('朴素贝叶斯，10次十折交叉验证结果:',sum(res)/len(res))

# 调用测试方法----------------------------------------------------------------------
testingNB()