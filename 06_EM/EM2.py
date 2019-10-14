# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 22:31:36 2019

@author: zsl
""" 
import numpy as np
from scipy.stats import multivariate_normal
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances_argmin
 
 
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
 
 
if __name__ == '__main__':
 
    iris = datasets.load_iris()
    data = iris.data
    y = iris.target
    
    feature_pairs = [[1, 3]]
#    feature_pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    iris_feature =['花萼长度','花萼宽度','花瓣长度','花瓣宽度']
    
#    re = []
    for k, pair in enumerate(feature_pairs):
        x = data[:, pair]
        #print(x)   # y是目标值的列向量 它等于分类0，1,2时的值对应的X的位置 就可以算出每一类的实际均值
        m = np.array([np.mean(x[y == i], axis=0) for i in range(3)])  # 均值的实际值
#        print ('实际均值 = \n', m)
        num_iter = 100
        n, d = x.shape
        mu1 = x.min(axis=0)
        mu2 = x.max(axis=0)
        mu3 = np.median(x,axis=0)
        print( mu1, mu2,mu3)
        sigma1 = np.identity(d)
        sigma2 = np.identity(d)
        sigma3 = np.identity(d)
        pi = 1.0/3
        # EM
        for i in range(num_iter):
            # E Step
            norm1 = multivariate_normal(mu1, sigma1)
            norm2 = multivariate_normal(mu2, sigma2)
            norm3 = multivariate_normal(mu3, sigma3)
            tau1 = pi * norm1.pdf(x)
            tau2 = pi * norm2.pdf(x)
            tau3 = pi * norm3.pdf(x)
            gamma1 = tau1 / (tau1 + tau2 + tau3)
            gamma2 = tau2 / (tau1 + tau2 + tau3)
            gamma3 = tau3 / (tau1 + tau2 + tau3)
            # M Step
            mu1 = np.dot(gamma1, x) / np.sum(gamma1)
            mu2 = np.dot(gamma2, x) / np.sum(gamma2)
            mu3 = np.dot(gamma3, x) / np.sum(gamma3)
            sigma1 = np.dot(gamma1 * (x - mu1).T, x - mu1) / np.sum(gamma1)
            sigma2 = np.dot(gamma2 * (x - mu2).T, x - mu2) / np.sum(gamma2)
            sigma3 = np.dot(gamma3 * (x - mu3).T, x - mu3) / np.sum(gamma3)
            pi = (np.sum(gamma1)+np.sum(gamma2)+np.sum(gamma3)) / n
#            print (i, ":\t", mu1, mu2)
#        print(u'类别概率:\t', pi)
#        print(u'均值:\t', mu1, mu2)
#        print(u'方差:\n', sigma1, '\n\n', sigma2, '\n')

        # 预测分类
        norm1 = multivariate_normal(mu1, sigma1)
        norm2 = multivariate_normal(mu2, sigma2)
        norm3 = multivariate_normal(mu3, sigma3)
        tau1 = norm1.pdf(x)
        tau2 = norm2.pdf(x)
        tau3 = norm3.pdf(x)
        
        y_hat=[]
        for i in range(y.size):
            if max(tau1[i],tau2[i],tau3[i])==tau1[i]:
                y_hat.append(0)
            elif max(tau1[i],tau2[i],tau3[i])==tau2[i]:
                y_hat.append(1)
            else:
                y_hat.append(2)
        
        fig = plt.figure(figsize=(10, 5), facecolor='w')
        ax = fig.add_subplot(121)
        ax.scatter(x[:, 0], x[:, 1],c='b', s=30, marker='o', edgecolors='k')
        ax.set_xlabel(iris_feature[pair[0]],fontsize=15)
        ax.set_ylabel(iris_feature[pair[1]],fontsize=15)
        ax.set_title(u'原始数据', fontsize=15)
        ax = fig.add_subplot(122)
        order = pairwise_distances_argmin(m,[mu1, mu2, mu3], axis=1,metric='euclidean')
        print (order)
        n_sample = y.size
        n_types = 3
        change = np.empty((n_types, n_sample), dtype=np.bool)
        for i in range(n_types):
            change[i] = y_hat == order[i]
        acc = u'准确率：%.2f%%' % (100*np.mean(y_hat == y))
        print (acc)
#        re.append(100*np.mean(y_hat == y))
        
        
        ax.scatter(x[change[0], 0], x[change[0], 1], c='r', s=30, marker='o', edgecolors='k')
        ax.scatter(x[change[1], 0], x[change[1], 1], c='g', s=30, marker='^', edgecolors='k')
        ax.scatter(x[change[2], 0], x[change[2], 1], c='#6060FF', s=30, marker='s', edgecolors='k')
        ax.set_xlabel(iris_feature[pair[0]],fontsize=15)
        ax.set_ylabel(iris_feature[pair[1]],fontsize=15)
        ax.set_title(u'EM算法分类', fontsize=15)
        plt.suptitle(u'EM算法的实现', fontsize=18)
        plt.subplots_adjust(top=0.90)
        plt.tight_layout()
        plt.show()
        
#        print(feature_pairs[re.index(max(re))])