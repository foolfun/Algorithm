# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 15:55:05 2019

@author: zsl
"""

import numpy as np
from sklearn.datasets import load_iris,load_wine
from sklearn import preprocessing
import matplotlib.pyplot as plt

def process_data(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.fit_transform(data)

#iris,wine
def loadData(data,y1):
    #load
    dataSet = data.data
    target = data.target
    #shuffle
    num_example=dataSet.shape[0]
    array = np.arange(num_example)
    np.random.shuffle(array)
    dataSet = dataSet[array]
    target = target[array]
    #transform
    classLabels = list(target)
    dataArr = process_data(dataSet)
    for i in range(len(classLabels)):
        if classLabels[i]==y1:
            classLabels[i]=1.0
        else:
            classLabels[i]=-1.0
    return dataArr,classLabels

#car
def loadDataSet(fileName):
    dataMat = []; classLabels = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split(',')
        # print(lineArr)
        if lineArr[0] == 'vhigh':
            lineArr[0] =1
        if lineArr[0] == 'high':
            lineArr[0] =2
        if lineArr[0] == 'med':
            lineArr[0] =3
        if lineArr[0] == 'low':
            lineArr[0] =4
        if lineArr[1] == 'vhigh':
            lineArr[1] =1
        if lineArr[1] == 'high':
            lineArr[1] =2
        if lineArr[1] == 'med':
            lineArr[1] =3
        if lineArr[1] == 'low':
            lineArr[1] =4
        if lineArr[2]=='2':
            lineArr[2]=1
        if lineArr[2]=='3':
            lineArr[2]=2
        if lineArr[2]=='4':
            lineArr[2]=3
        if lineArr[2]=='5more':
            lineArr[2]=4
        if lineArr[3]=='2':
            lineArr[3]=1
        if lineArr[3]=='4':
            lineArr[3]=2
        if lineArr[3]=='more':
            lineArr[3]=3
        if lineArr[4]=='small':
            lineArr[4]=1
        if lineArr[4]=='med':
            lineArr[4]=2
        if lineArr[4]=='big':
            lineArr[4]=3
        if lineArr[5]=='low':
            lineArr[5]=1
        if lineArr[5]=='med':
            lineArr[5]=2
        if lineArr[5]=='high':
            lineArr[5]=3
        dataMat.append([float(lineArr[0]),float(lineArr[1]), float(lineArr[2]),
                        float(lineArr[3]),float(lineArr[4]),
                        float(lineArr[5])])
       

        if lineArr[6] == 'unacc':
            lineArr[6] =1.0
#        elif lineArr[6] == 'acc':
#            lineArr[6] =1
#        elif lineArr[6] == 'good':
#            lineArr[6] =2
        else:
            lineArr[6] =-1.0

        classLabels.append(float(lineArr[6]))
        
    return np.array(dataMat,dtype="float64"),classLabels
    
def stumpClassify(dataMartix,col,threshVal,threshIneq):
    retArray = np.ones((np.shape(dataMartix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMartix[:,col] <= threshVal] = -1.0
    else:
        retArray[dataMartix[:,col] > threshVal] = -1.0
    return retArray

def buildStump(dataArr,classLabels,W):
    dataMatrix = np.mat(dataArr)
    classMatrix = np.mat(classLabels).T
    m,n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump={}
    bestClasEst = np.mat(np.zeros((m,1)))
    minError = np.inf
    for i in range(n):
        colMax = dataMatrix[:,i].max()
        colMin = dataMatrix[:,i].min()
        stepSize = (colMax-colMin)/numSteps
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:
                threshVal = (colMin +float(j)*stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
#                print('predictedVals:=================',predictedVals)
                errArr = np.mat(np.ones((m,1)))
#                print('errArr',errArr)
                errArr[predictedVals == classMatrix] = 0
                weightedError = W.T*errArr
#                print("split: col %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['col']=i
                    bestStump['thresh']=threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst

def adaBoostTrainDS(dataArr,classLables,T=30):
    weakClassArr = []
    errorList =[]
    m = np.shape(dataArr)[0]
    W = np.mat(np.ones((m,1))/m)
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(T):
        bestStump,error,classEst=buildStump(dataArr,classLables,W)
        alpha = float(0.5*np.log((1.0-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        expon = np.multiply(-1*alpha*np.mat(classLables).T,classEst)
        W = np.multiply(W,np.exp(expon))
        W = W/W.sum()
        aggClassEst += alpha*classEst
#        print(aggClassEst)
        aggErrors = np.multiply(np.sign(aggClassEst)!=np.mat(classLables).T,np.ones((m,1)))
#        print('==============',aggErrors)
        errorRate = aggErrors.sum()/m
#        print('total error',errorRate)
        errorList.append(errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr,aggClassEst,errorList

def adaClassify(datToClass,classifierArr):
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['col'],
                                 classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
#        print(np.sign(aggClassEst))
    return np.sign(aggClassEst)

def classifytest(testDataSet, classifierArr,target_test):
    """
        计算准确率
    """
    i =0
    cnt = 0
    for testVec in testDataSet:
        pre =adaClassify(testDataSet,classifierArr)
        if (pre == target_test[i]).all():
            cnt +=1
        i += 1
#    print('cnt',cnt)
    return cnt/len(target_test)

##iris
data = load_iris()
dataArr1,classLabels1=loadData(data,0)

#wine
data = load_wine()
dataArr2,classLabels2=loadData(data,3)

##car
dataArr3,classLabels3=loadDataSet('car.data')
#W = np.mat(np.ones((np.shape(dataArr)[0],1))/np.shape(dataArr)[0])
#bestStump,minError,bestClasEst=buildStump(dataArr,classLabels,W)
#classifierArr,aggClassEst,errorList = adaBoostTrainDS(dataArr1,classLabels1,50)
 
num_example = dataArr2.shape[0]
sample = np.int(num_example * 0.9)
x_train = dataArr2[: sample]
y_train = classLabels2[: sample]
x_test = dataArr2[sample:]
y_test = classLabels2[sample:]
res=[]
for j in range(10):
    pri = []
    tmp = 0
    prob = 0.1
    for i in range(10):
        classifierArr,aggClassEst,errorList = adaBoostTrainDS(x_train[tmp:np.int(sample*prob)],y_train[tmp:np.int(sample*prob)],50)
        pre=classifytest(x_test, classifierArr,y_test)
        pri.append(pre)
        tmp=np.int(sample*prob)+1
        prob += 0.1
    res.append(sum(pri)/len(pri))
    print(sum(pri)/len(pri))
print('wine AdaBoost，10次十折交叉验证结果:',(sum(res)/len(res)))



#plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
#plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#plt.figure()
#ln1, = plt.plot(errorList1,linestyle='dashed',linewidth=0.5,color='red',marker='.',)
#ln2, = plt.plot(errorList2,linestyle='dashed',linewidth=0.5,color='b',marker='.',)
#ln3, = plt.plot(errorList3,linestyle='dashed',linewidth=0.5,color='g',marker='.',)
#ln4, = plt.plot(errorList4,linestyle='dashed',linewidth=0.5,color='yellow',marker='.',)
#plt.ylim(0,0.3)
#plt.legend(handles=[ln1,ln2,ln3,ln4], labels=['unacc', 'acc','good','vgood'],
#    loc='uper right')
#plt.ylabel('errorRate')
#plt.title('训练50次过程中数据集car错误率的变化情况')
#plt.show()