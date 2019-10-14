# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 22:31:36 2019

@author: zsl
""" 
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances_argmin
 
 
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

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

 
if __name__ == '__main__':
 
 
    data,y = loadDataSet('car.data')
    data = np.array(data)
    y = np.array(y)
    
    k = []
    for i in range(6):
        j=i+1
        while j<7:
            k.append([i,j])
            j=j+1
    
    feature_pairs = [[0, 1],
 [0, 2],
 [0, 3],
 [0, 4],
 [0, 5],
 [0, 6],
 [1, 2],
 [1, 3],
 [1, 4],
 [1, 5],
 [1, 6],
 [2, 3],
 [2, 4],
 [2, 5],
 [2, 6],
 [3, 4],
 [3, 5],
 [3, 6],
 [4, 5],
 [4, 6],
 [5, 6]]
    iris_feature =['buying','maint','doors','persons',' lug_boot','safety']
    re=[]
    for k, pair in enumerate(feature_pairs):
        x = data[:, pair]
        #print(x)   # y是目标值的列向量 它等于分类0，1,2时的值对应的X的位置 就可以算出每一类的实际均值
        m = np.array([np.mean(x[y == i], axis=0) for i in range(4)])  # 均值的实际值
#        print ('实际均值 = \n', m)
        num_iter = 100
        n, d = x.shape
        mu1 = x.min(axis=0)
        mu2 = x.max(axis=0)
        mu3 = np.median(x,axis=0)
        mu4 = np.median(x,axis=0)
        print( mu1, mu2,mu3,mu4)
        sigma1 = np.identity(d)
        sigma2 = np.identity(d)
        sigma3 = np.identity(d)
        sigma4 = np.identity(d)
        pi = 1.0/4
        # EM
        for i in range(num_iter):
            # E Step
            norm1 = multivariate_normal(mu1, sigma1)
            norm2 = multivariate_normal(mu2, sigma2)
            norm3 = multivariate_normal(mu3, sigma3)
            norm4 = multivariate_normal(mu4, sigma4)
            tau1 = pi * norm1.pdf(x)
            tau2 = pi * norm2.pdf(x)
            tau3 = pi * norm3.pdf(x)
            tau4 = pi * norm4.pdf(x)
            gamma1 = tau1 / (tau1 + tau2 + tau3+tau4)
            gamma2 = tau2 / (tau1 + tau2 + tau3+tau4)
            gamma3 = tau3 / (tau1 + tau2 + tau3+tau4)
            gamma4 = tau4 / (tau1 + tau2 + tau3+tau4)
            # M Step
            mu1 = np.dot(gamma1, x) / np.sum(gamma1)
            mu2 = np.dot(gamma2, x) / np.sum(gamma2)
            mu3 = np.dot(gamma3, x) / np.sum(gamma3)
            mu4 = np.dot(gamma4, x) / np.sum(gamma4)
            sigma1 = np.dot(gamma1 * (x - mu1).T, x - mu1) / np.sum(gamma1)
            sigma2 = np.dot(gamma2 * (x - mu2).T, x - mu2) / np.sum(gamma2)
            sigma3 = np.dot(gamma3 * (x - mu3).T, x - mu3) / np.sum(gamma3)
            sigma4 = np.dot(gamma4 * (x - mu4).T, x - mu4) / np.sum(gamma4)
            pi = (np.sum(gamma1)+np.sum(gamma2)+np.sum(gamma3)+np.sum(gamma4)) / n
#            print (i, ":\t", mu1, mu2)
#        print(u'类别概率:\t', pi)
#        print(u'均值:\t', mu1, mu2)
#        print(u'方差:\n', sigma1, '\n\n', sigma2, '\n')

        # 预测分类
        norm1 = multivariate_normal(mu1, sigma1)
        norm2 = multivariate_normal(mu2, sigma2)
        norm3 = multivariate_normal(mu3, sigma3)
        norm4 = multivariate_normal(mu4, sigma4)
        tau1 = norm1.pdf(x)
        tau2 = norm2.pdf(x)
        tau3 = norm3.pdf(x)
        tau4 = norm4.pdf(x)
        y_hat=[]
        for i in range(y.size):
            if max(tau1[i],tau2[i],tau3[i],tau4[i])==tau1[i]:
                y_hat.append(0)
            elif max(tau1[i],tau2[i],tau3[i],tau4[i])==tau2[i]:
                y_hat.append(1)
            elif max(tau1[i],tau2[i],tau3[i],tau4[i])==tau3[i]:
                y_hat.append(2)
            else:
                y_hat.append(3)
        
        
        order = pairwise_distances_argmin(m,[mu1, mu2, mu3, mu4], axis=1,metric='euclidean')
        print (order)
        n_sample = y.size
        n_types = 4
        change = np.empty((n_types, n_sample), dtype=np.bool)
        for i in range(n_types):
            change[i] = y_hat == order[i]
        acc = u'准确率：%.2f%%' % (100*np.mean(y_hat == y))
        print (acc)
        re.append(100*np.mean(y_hat == y))
        
        fig = plt.figure(figsize=(10, 5), facecolor='w')
        ax = fig.add_subplot(121)
        ax.scatter(x[:, 0], x[:, 1],c='b', s=30, marker='o', edgecolors='k')
        ax.set_xlabel(iris_feature[pair[0]],fontsize=15)
        ax.set_ylabel(iris_feature[pair[1]],fontsize=15)
        ax.set_title(u'原始数据', fontsize=15)
        ax = fig.add_subplot(122)
        ax.scatter(x[change[0], 0], x[change[0], 1], c='r', s=30, marker='o', edgecolors='k')
        ax.scatter(x[change[1], 0], x[change[1], 1], c='g', s=30, marker='^', edgecolors='k')
        ax.scatter(x[change[2], 0], x[change[2], 1], c='#6060FF', s=30, marker='s', edgecolors='k')
        ax.scatter(x[change[3], 0], x[change[3], 1], c='#D2691E', s=30, marker='.', edgecolors='k')
        ax.set_xlabel(iris_feature[pair[0]],fontsize=15)
        ax.set_ylabel(iris_feature[pair[1]],fontsize=15)
        ax.set_title(u'EM算法分类', fontsize=15)
        plt.suptitle(u'EM算法的实现', fontsize=18)
        plt.subplots_adjust(top=0.90)
        plt.tight_layout()
        plt.show()