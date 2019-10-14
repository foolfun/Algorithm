# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 16:34:12 2019

@author: zsl
"""

import numpy as np

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris,load_wine
from sklearn.datasets.samples_generator import make_blobs
from sklearn import preprocessing
from sklearn.cluster import Birch


#iris

#
#data=load_iris()
#dataset = data.data
#y = data.target
#X=dataset

##wine
data=load_wine()
dataset = data.data
y = data.target
X=dataset

##设置birch函数

birch = Birch(n_clusters= None)

##训练数据

y_pred =birch.fit_predict(X)

##绘图

plt.figure()

plt.subplot(2,2,1)

plt.scatter(X[:,0],X[:,1])

plt.title('DataSample')

plt.subplot(2,2,2)

plt.scatter(X[:,0], X[:, 1], c=y_pred)

plt.title('None')
##设置birch函数

birch =Birch(n_clusters = 3)

##训练数据

y_pred =birch.fit_predict(X)

plt.subplot(2,2,3)

plt.scatter(X[:,0], X[:, 1], c=y_pred)

plt.title('n_clusters=3')

plt.show()