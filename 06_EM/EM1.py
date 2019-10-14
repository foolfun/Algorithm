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
 
    wine = datasets.load_wine()

    feature_name = ['酒精','苹果酸','灰','灰的碱性','镁','总酚','类黄酮','非黄烷类酚类','花青素','颜色强度','色调','od280/od315稀释葡萄酒','脯氨酸']
    
    # and testing (25%) sets.
    skf = StratifiedKFold(n_splits=4)#分层采样，确保训练集，测试集中各类别样本的比例与原始数据集中相同。
    # Only take the first fold.
    train_index, test_index = next(iter(skf.split(wine.data, wine.target)))
    data = wine.data
    
    def maxminnorm(array):
        maxcols=array.max(axis=0)
        mincols=array.min(axis=0)
        data_shape = array.shape
        data_rows = data_shape[0]
        data_cols = data_shape[1]
        t=np.empty((data_rows,data_cols))
        for i in range(data_cols):
            t[:,i]=(array[:,i]-mincols[i])/(maxcols[i]-mincols[i])
        return t
    
    data=maxminnorm(data)
    
    y = wine.target

    
#    k = []
#    for i in range(12):
#        j=i+1
#        while j<13:
#            k.append([i,j])
#            j=j+1
    
#    feature_pairs = [[0, 1],
# [0, 2],
# [0, 3],
# [0, 4],
# [0, 5],
# [0, 6],
# [0, 7],
# [0, 8],
# [0, 9],
# [0, 10],
# [0, 11],
# [0, 12],
# [1, 2],
# [1, 3],
# [1, 4],
# [1, 5],
# [1, 6],
# [1, 7],
# [1, 8],
# [1, 9],
# [1, 10],
# [1, 11],
# [1, 12],
# [2, 3],
# [2, 4],
# [2, 5],
# [2, 6],
# [2, 7],
# [2, 8],
# [2, 9],
# [2, 10],
# [2, 11],
# [2, 12],
# [3, 4],
# [3, 5],
# [3, 6],
# [3, 7],
# [3, 8],
# [3, 9],
# [3, 10],
# [3, 11],
# [3, 12],
# [4, 5],
# [4, 6],
# [4, 7],
# [4, 8],
# [4, 9],
# [4, 10],
# [4, 11],
# [4, 12],
# [5, 6],
# [5, 7],
# [5, 8],
# [5, 9],
# [5, 10],
# [5, 11],
# [5, 12],
# [6, 7],
# [6, 8],
# [6, 9],
# [6, 10],
# [6, 11],
# [6, 12],
# [7, 8],
# [7, 9],
# [7, 10],
# [7, 11],
# [7, 12],
# [8, 9],
# [8, 10],
# [8, 11],
# [8, 12],
# [9, 10],
# [9, 11],
# [9, 12],
# [10, 11],
# [10, 12],
# [11, 12]]
#    re = []
    feature_pairs=[[3,11]]
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
        
        fig = plt.figure(figsize=(10, 5), facecolor='w')
        ax = fig.add_subplot(121)
        ax.scatter(x[:, 0], x[:, 1],c='b', s=30, marker='o', edgecolors='k')
        ax.set_xlabel(feature_name[pair[0]],fontsize=15)
        ax.set_ylabel(feature_name[pair[1]],fontsize=15)
        ax.set_title(u'原始数据', fontsize=15)
        ax = fig.add_subplot(122)
        ax.scatter(x[change[0], 0], x[change[0], 1], c='r', s=30, marker='o', edgecolors='k')
        ax.scatter(x[change[1], 0], x[change[1], 1], c='g', s=30, marker='^', edgecolors='k')
        ax.scatter(x[change[2], 0], x[change[2], 1], c='#6060FF', s=30, marker='s', edgecolors='k')
        ax.set_xlabel(feature_name[pair[0]],fontsize=15)
        ax.set_ylabel(feature_name[pair[1]],fontsize=15)
        ax.set_title(u'EM算法聚类', fontsize=15)
        plt.suptitle(u'EM算法的实现', fontsize=18)
        plt.subplots_adjust(top=0.90)
        plt.tight_layout()
        plt.show()
        
#        print(feature_pairs[re.index(max(re))])
        