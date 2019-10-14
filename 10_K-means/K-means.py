# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 16:38:52 2019

@author: zsl
"""
from numpy import *
from sklearn.datasets import load_iris,load_wine
from sklearn import preprocessing
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def calDistance(a,b):
    return sqrt(sum(power(a-b,2)))

def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j]) 
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids

def kMeans(dataSet, k, distMeas=calDistance, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    SSE=[]
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex: 
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        print(centroids)
        SSE.append(sum(clusterAssment[:,1]))
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:] = mean(ptsInClust, axis=0) 
    return centroids, clusterAssment,SSE

def process_data(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.fit_transform(dataset)

def plot_scatter(feature_name,dataset,target,mycentroids,myclusterAssment):
    x=[];y=[]
    n=shape(mycentroids)[0]
    for cent in range(n):
        x.append(dataset[nonzero(myclusterAssment[:,0].A==cent)[0]])
    
    for cent in range(n):
        y.append(dataset[nonzero(target==cent)[0]])
    
    fig = plt.figure(figsize=(10, 5), facecolor='w')
    ax = fig.add_subplot(121)
    ax.scatter(y[0][:,0], y[0][:,1], c='r', s=30, marker='o', edgecolors='k')
    ax.scatter(y[1][:,0], y[1][:,1], c='g', s=30, marker='^', edgecolors='k')
    ax.scatter(y[2][:,0], y[2][:,1], c='#6060FF', s=30, marker='s', edgecolors='k')
    
    ax.set_xlabel(feature_name[0],fontsize=15)
    ax.set_ylabel(feature_name[1],fontsize=15)
    ax.set_title(u'origin', fontsize=15)
    ax = fig.add_subplot(122)
    ax.scatter(x[0][:,0], x[0][:,1], c='r', s=30, marker='o', edgecolors='k')
    ax.scatter(x[1][:,0], x[1][:,1], c='g', s=30, marker='^', edgecolors='k')
    ax.scatter(x[2][:,0], x[2][:,1], c='#6060FF', s=30, marker='s', edgecolors='k')
    ax.set_xlabel(feature_name[0],fontsize=15)
    ax.set_ylabel(feature_name[1],fontsize=15)
    ax.set_title(u'K-means', fontsize=15)
    plt.tight_layout()
    plt.show()
    
#iris
data=load_iris()
dataset = data.data
target = data.target
dataset=process_data(dataset)
datMat=mat(dataset)
mycentroids,myclusterAssment,sse1=kMeans(datMat,3)
plt.plot(sse1)
#7.13
feature_name =['Calyx length','Calyx width']
plot_scatter(feature_name,dataset,target,mycentroids,myclusterAssment)
    
    
##wine
#data=load_wine()
#dataset = data.data
#target = data.target
#dataset=process_data(dataset)
#df = pd.DataFrame(dataset)
#dfcorr=df.corr()
##plot heatmap
#plt.subplots(figsize=(13, 13)) 
#sns.heatmap(dfcorr, annot=True, vmax=1, square=True, 
#            cmap="Blues")
#plt.savefig('d:/heatmap.png')
#plt.show()
##01245
#df = df[[0,1,2,4,5]]
#datMat=mat(df.values)
#mycentroids,myclusterAssment,sse2=kMeans(datMat,3)
##16.90
#plt.plot(sse2)
#feature_name =['alcohol','malic acid']
#plot_scatter(feature_name,dataset,target,mycentroids,myclusterAssment)
 
##car
def loadDataSet(fileName):
    dataMat = []; labelMat = []
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
            lineArr[6] =0
        elif lineArr[6] == 'acc':
            lineArr[6] =1
        elif lineArr[6] == 'good':
            lineArr[6] =2
        else:
            lineArr[6] =3

        labelMat.append(float(lineArr[6]))
  
    return dataMat,labelMat

#dataset,target = loadDataSet('car.data')
#target=int32(target)
#dataset = array(dataset)
#datMat=mat(dataset)
#mycentroids,myclusterAssment,sse3=kMeans(datMat,4)
##5937.575268319763
#plt.plot(sse3)
#x=[];y=[]
#n=shape(mycentroids)[0]
#for cent in range(n):
#    x.append(dataset[nonzero(myclusterAssment[:,0].A==cent)[0]])
#
#for cent in range(n):
#    y.append(dataset[nonzero(target==cent)[0]])
#
#fig = plt.figure(figsize=(10, 5), facecolor='w')
#ax = fig.add_subplot(121)
#ax.scatter(y[0][:,0], y[0][:,1], c='r', s=30, marker='o', edgecolors='k')
#ax.scatter(y[1][:,0], y[1][:,1], c='g', s=30, marker='^', edgecolors='k')
#ax.scatter(y[2][:,0], y[2][:,1], c='#6060FF', s=30, marker='s', edgecolors='k')
#ax.scatter(y[3][:,0], y[3][:,1], c='gold', s=30, marker='s', edgecolors='k')
#feature_name=['buying','maint']     
#ax.set_xlabel(feature_name[0],fontsize=15)
#ax.set_ylabel(feature_name[1],fontsize=15)
#ax.set_title(u'origin', fontsize=15)
#ax = fig.add_subplot(122)
#ax.scatter(x[0][:,0], x[0][:,1], c='r', s=30, marker='o', edgecolors='k')
#ax.scatter(x[1][:,0], x[1][:,1], c='g', s=30, marker='^', edgecolors='k')
#ax.scatter(x[2][:,0], x[2][:,1], c='#6060FF', s=30, marker='s', edgecolors='k')
#ax.scatter(x[3][:,0], x[3][:,1], c='gold', s=30, marker='s', edgecolors='k')
#ax.set_xlabel(feature_name[0],fontsize=15)
#ax.set_ylabel(feature_name[1],fontsize=15)
#ax.set_title(u'K-means', fontsize=15)
#plt.tight_layout()
#plt.show()
    

ln1, = plt.plot(sse1, color = 'red', linewidth = 2.0, linestyle = '--')
ln2, = plt.plot(sse2, color = 'blue', linewidth = 2.0, linestyle = '--')
ln3, = plt.plot(sse3, color = 'pink', linewidth = 2.0, linestyle = '--')
plt.legend(handles=[ln1,ln2,ln3], labels=['iris', 'wine','car'],
    loc='uper right')

