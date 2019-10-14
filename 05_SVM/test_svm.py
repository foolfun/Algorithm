# -*- coding: utf-8 -*-
"""
Created on Thu May 23 21:56:00 2019

@author: zsl
"""
from sklearn import datasets
import numpy as np
import SVM

################## test svm #####################
## step 1: load data
print ("step 1: load data...")
iris = datasets.load_iris()
wine = datasets.load_wine()
x,y = iris.data,iris.target
# 使数据集乱序
num_example = x.shape[0]
array = np.arange(num_example)
np.random.shuffle(array)
dataset = x[array]
target = y[array]
for i in range(len(target)):
        if target[i] == 2:
            target[i] = 0
# 训练集: 验证集 = 9: 1
sample = np.int(num_example * 0.9)
x_train = dataset[: sample]
y_train = target[: sample]
x_test = dataset[sample:]
y_test = target[sample:]

## step 2: training...
print("step 2: training...")
C = 0.6
toler = 0.001
maxIter = 50
svmClassifier = SVM.trainSVM(x_train, y_train, C, toler, maxIter, kernelOption = ('linear', 0))

## step 3: testing
print ("step 3: testing...")
accuracy = SVM.testSVM(svmClassifier, x_test, y_test)

## step 4: show the result
print( "step 4: show the result..."	)
print( 'The classify accuracy is: %.3f%%' % (accuracy * 100))
SVM.showSVM(svmClassifier)