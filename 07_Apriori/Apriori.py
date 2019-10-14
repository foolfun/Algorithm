# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 22:54:52 2019

@author: zsl
"""

from numpy import *
import numpy as np
from sklearn.datasets import load_iris,load_wine
import pandas as pd
# 构造数据
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split(',')
        # print(lineArr)
        if lineArr[0] == 'vhigh':
            lineArr[0] ='a1'
        if lineArr[0] == 'high':
            lineArr[0] ='a2'
        if lineArr[0] == 'med':
            lineArr[0] ='a3'
        if lineArr[0] == 'low':
            lineArr[0] ='a4'
            
        if lineArr[1] == 'vhigh':
            lineArr[1] ='b1'
        if lineArr[1] == 'high':
            lineArr[1] ='b2'
        if lineArr[1] == 'med':
            lineArr[1] ='b3'
        if lineArr[1] == 'low':
            lineArr[1] ='b4'
            
        if lineArr[2]=='2':
            lineArr[2]='c1'
        if lineArr[2]=='3':
            lineArr[2]='c2'
        if lineArr[2]=='4':
            lineArr[2]='c3'
        if lineArr[2]=='5more':
            lineArr[2]='c4'
            
        if lineArr[3]=='2':
            lineArr[3]='d1'
        if lineArr[3]=='4':
            lineArr[3]='d2'
        if lineArr[3]=='more':
            lineArr[3]='d3'
            
        if lineArr[4]=='small':
            lineArr[4]='e1'
        if lineArr[4]=='med':
            lineArr[4]='e2'
        if lineArr[4]=='big':
            lineArr[4]='e3'
            
        if lineArr[5]=='low':
            lineArr[5]='f1'
        if lineArr[5]=='med':
            lineArr[5]='f2'
        if lineArr[5]=='high':
            lineArr[5]='f3'
      
        dataMat.append(lineArr)
  
    return dataMat
 
# 将所有元素转换为frozenset型字典，存放到列表中
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    # 使用frozenset是为了后面可以将这些值作为字典的键
    return list(map(frozenset, C1))  # frozenset一种不可变的集合，set可变集合
 
# 过滤掉不符合支持度的集合
# 返回 频繁项集列表retList 所有元素的支持度字典
def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):   # 判断can是否是tid的《子集》 （这里使用子集的方式来判断两者的关系）
                if can not in ssCnt:    # 统计该值在整个记录中满足子集的次数（以字典的形式记录，frozenset为键）
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []        # 重新记录满足条件的数据值（即支持度大于阈值的数据）
    supportData = {}    # 每个数据值的支持度
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData # 排除不符合支持度元素后的元素 每个元素支持度
 
# 生成所有可以组合的集合
# 频繁项集列表Lk 项集元素个数k  [frozenset({2, 3}), frozenset({3, 5})] -> [frozenset({2, 3, 5})]
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk): # 两层循环比较Lk中的每个元素与其它元素
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]  # 将集合转为list后取值
            L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()        # 这里说明一下：该函数每次比较两个list的前k-2个元素，如果相同则求并集得到k个元素的集合
            if L1==L2:
                retList.append(Lk[i] | Lk[j]) # 求并集
    return retList  # 返回频繁项集列表Ck
 
# 封装所有步骤的函数
# 返回 所有满足大于阈值的组合 集合支持度列表
def apriori(dataSet, minSupport = 0.5):
    D = list(map(set, dataSet)) # 转换列表记录为字典  [{1, 3, 4}, {2, 3, 5}, {1, 2, 3, 5}, {2, 5}]
    C1 = createC1(dataSet)      # 将每个元素转会为frozenset字典    [frozenset({1}), frozenset({2}), frozenset({3}), frozenset({4}), frozenset({5})]
    L1, supportData = scanD(D, C1, minSupport)  # 过滤数据
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):    # 若仍有满足支持度的集合则继续做关联分析
        Ck = aprioriGen(L[k-2], k)  # Ck候选频繁项集
        Lk, supK = scanD(D, Ck, minSupport) # Lk频繁项集
        supportData.update(supK)    # 更新字典（把新出现的集合:支持度加入到supportData中）
        L.append(Lk)
        k += 1  # 每次新组合的元素都只增加了一个，所以k也+1（k表示元素个数）
    return L, supportData

#dataSet = loadDataSet('car.data')
#L,suppData = apriori(dataSet)
#print(L)
#print(suppData)
# 获取关联规则的封装函数
def generateRules(L, supportData, minConf=0.7):  # supportData 是一个字典
    bigRuleList = []
    for i in range(1, len(L)):  # 从为2个元素的集合开始
        for freqSet in L[i]:
            # 只包含单个元素的集合列表
            H1 = [frozenset([item]) for item in freqSet]    # frozenset({2, 3}) 转换为 [frozenset({2}), frozenset({3})]
            # 如果集合元素大于2个，则需要处理才能获得规则
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf) # 集合元素 集合拆分后的列表 。。。
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

# 对规则进行评估 获得满足最小可信度的关联规则
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = []  # 创建一个新的列表去返回
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq]  # 计算置信度
        if conf >= minConf:
            print(freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH
 
# 生成候选规则集合
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)): # 尝试进一步合并
        Hmp1 = aprioriGen(H, m+1) # 将单个集合元素两两合并
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):    #need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


#iris
#data=load_iris()
#dataSet = data.data
#labels = data.feature_names
#target = data.target
#r=[]
#for i in target:
#    if i==0:
#        i='Iris-setosa'
#    elif i==1:
#        i='Iris-versicolor'
#    else:
#        i='Iris-virginica'
#    r.append(i)
#data = []
#for i in range(4):
#    tmp=pd.cut(dataSet[:,1],3,labels=['a'+str(i),'b'+str(i),'c'+str(i)])
#    data.append(tmp)
#data.append(r)
#data = np.array(list(zip(*data)))

#wine
#data=load_wine()
#dataSet = data.data
#labels = data.feature_names
#r = data.target
#data = []
#for i in range(13):
#    tmp=pd.cut(dataSet[:,1],3,labels=['a'+str(i),'b'+str(i),'c'+str(i)])
#    data.append(tmp)
#data.append(r)
#data = np.array(list(zip(*data)))
#dataSet = column_stack((dataSet,target))

#car
data = loadDataSet('car.data')


L,suppData = apriori(data,minSupport=0.3)
rules = generateRules(L,suppData,minConf=0.4)
# rules = generateRules(L,suppData,minConf=0.5)
#print(rules)
rules1=sorted(rules,key=lambda x:x[2],reverse=True)
conf_list =[]
name_list=[]
res=[]
if len(rules)>3:
    for i in range(4):
        conf_list.append(rules1[i][2])
        name_list.append((rules1[i][0],'-->',rules1[i][1]))
        res.append(rules1[i])
    import matplotlib.pyplot as plt 
    num_list = conf_list
    rects=plt.bar(range(len(num_list)), num_list, color='rgby')
    # X轴标题
    index=[0,1,2,3]
    index=[float(c)+0.4 for c in index]
    plt.ylim(ymax=1, ymin=0)
    plt.xticks(index, name_list)
    plt.ylabel("conf(%)") #X轴标签
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, str(height)+'%', ha='center', va='bottom')
    plt.show()
else:
    print(rules)
    
