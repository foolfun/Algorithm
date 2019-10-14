# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 01:33:54 2019

@author: zsl
"""

import numpy as np
import pandas as pd
def loadSample():
    x = np.mat( '1,1,2,2;\
                 1,2,1,2\
                 ').T
    x = np.array(x)
    y = np.mat('0,1,1,0').T
    return x,y

def loadData(fileName):
    df = pd.read_csv(fileName)
    # 对数值型属性离散处理
    for h in df.columns[:-1]:
        if df.dtypes[h] in ['float64', 'int64']: 
            df[h] = pd.cut(df[h],3,labels=[h+'_低',h+'_中',h+'_高'])
    return df.values
# 条件熵 H(Y|X)=H(X,Y) - H(X)
from math import log
def informEnt(dataMat):
    m = np.shape(dataMat)[0]
    count ={}
    hxy =0.
    for data in dataMat:
        st = str(data)
        if st not in count.keys(): count[st] =0.
        count[st] +=1.
#     print(count)
    for key in count.keys():
        p = count[key]/m
        hxy -= p*log(p,2)
#         print(hxy)
    return hxy

def splitDataMat(d1, d2):#拼接数据集
    if len(d1)==0: return d2#d1又是为空
    d1 = np.mat(d1).T
    d2 = np.mat(d2).T
    print(np.shape(d1),np.shape(d2))
    return np.hstack((d1,d2))

# 求核心属性
def core(dataMat):
    d = np.shape(dataMat)[1]
    retArr = np.zeros(d)
    y = [[i] for i in dataMat[:,-1]]
    # 计算H(Y|X)=H(X,Y) - H(X)
    HXY = informEnt(dataMat) #H(X,Y)
    HX = informEnt(dataMat[:,:-1]) #H(X)
    HY_X = HXY - HX #H(Y|X)
    print("H(Y|X)=",HY_X)
    
    for c in range(d):
        X_c = np.hstack((dataMat[:,:c], dataMat[:,c:-1]))
        HX_c = informEnt(X_c)
#         print(np.shape(X_c),np.shape(y))
        YX_c = np.hstack((X_c, y))
        HYX_c = informEnt(YX_c)
        
        HY_X_c = HYX_c - HX_c
#         print(HY_X ,HY_X_c)
        if HY_X < HY_X_c: retArr[c] =1
    return retArr
def reduct(dataMat, COFlat):
    m,d = np.shape(dataMat)
    y = [[i] for i in dataMat[:,-1]]
    # 计算H(Y|X)=H(X,Y) - H(X)
    HXY = informEnt(dataMat) #H(X,Y)
    HX = informEnt(dataMat[:,:-1]) #H(X)
    HY_X = HXY - HX #H(Y|X)
    
    CORE = np.arange(d)[COFlat ==1].tolist()#0-未遍历，1-核心属性, 2-冗余属性
    CO =[]#核心数据
    for c in CORE:
        CO = splitData(CO, dataMat[c],m)
    HY_CO =informEnt(dataMat[:,-1]) -informEnt(CO)#这里默认CORE为空
    # 计算重要度 SGF=H(Y|CO) - H(Y|CO+c), SGF越大越重要
    while HY_CO != HY_X:#最大迭代次数为100
        print('H(Y|CO)=',HY_CO)
        maxSGF =-1.
        maxHY_COc =0.
        maxCol = -1
        for c in range(d-1):#遍历所有属性
            if COFlat[c] != 0: continue
            C = [[i] for i in dataMat[:,c]]
            # 计算重要度
            if len(CO)==0: COc = C
            else: COc = np.hstack((CO, C))
            HCOc = informEnt(COc)
#             print('  ',c,HCOc)
            YCOc = np.hstack((COc, y))
            HYCOc= informEnt(YCOc)
#             print('  ',c,HYCOc)
            HY_COc = HYCOc - HCOc
            SGF = HY_CO - HY_COc
            print('  ',c,SGF)
            if SGF <=0: COFlat[c]=2;print('     多余属性',c)
#             print("c:%d, HCOc:%.3f, HYCOc:%.3f, HY_COc：%.3f, SGF:%.3f"%(c,HCOc,HYCOc,HY_COc,SGF ))
            # 更新最大值
            elif maxSGF < SGF: 
                maxSGF = SGF
                maxHY_COc = HY_COc
                maxCol = c  
        # 更新核心属性
        if  maxCol ==-1: break
        COFlat[maxCol] =1;CORE.append(maxCol)
        C = [[i] for i in dataMat[:,maxCol]]
        if len(CO)==0: CO = C
        else: CO = np.hstack((CO, C))
#         print(CO)
        HY_CO = maxHY_COc
        print("  maxCol:%s, maxSGF:%.3f, "%(maxCol,maxSGF))
    return CORE
dm = loadData("D:/summer2019/datasets/iris.csv")
COFlat = core(dm)
print('核心属性：',COFlat)
CORE = reduct(dm, COFlat)
print('约简后的核心属性：',CORE)