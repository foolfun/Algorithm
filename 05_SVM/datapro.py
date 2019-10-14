# -*- coding: utf-8 -*-
"""
Created on Thu May 23 21:58:22 2019

@author: zsl
"""
from sklearn import datasets
'''
car
'''
iris = datasets.load_iris()
wine = datasets.load_wine()
X, y = iris.data, iris.target

dataSet = []
labels = []
fileIn = open('car.data')
for line in fileIn.readlines():
	lineArr = line.strip().split('\t')
	dataSet.append([float(lineArr[0]), float(lineArr[1])])
	labels.append(float(lineArr[2]))
