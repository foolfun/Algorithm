# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 18:40:31 2019

@author: zsl
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris,load_wine
from sklearn import preprocessing

def process_data(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.fit_transform(data)
#iris,wine
def loadData(data):
    #load
    dataSet = data.data
    target = data.target
    #shuffle
    num_example=dataSet.shape[0]
    array = np.arange(num_example)
    np.random.shuffle(array)
    dataSet = process_data(dataSet)
    dataSet = dataSet[array]
    target = target[array]
   
    return dataSet,target

##iris
#data = load_iris()
#data, label=loadData(data)

#wine
data = load_wine()
data, label=loadData(data)

print('=====read===ok======')

num_example = data.shape[0]
# 训练集: 验证集 = 7: 3, 考虑到样本较少，验证集的结果可以反映测试集结果
sample = np.int(num_example * 0.7)
x_train = data[: sample]
y_train = label[: sample]
x_test = data[sample:]
y_test = label[sample:]

#激活函数
def sigmoid(x):
    if type(x)!=np.ndarray:
       return 1/(1+math.exp(-x))
    return 1/(1+np.exp(-x))

#激活函数的偏导数
def sigDer(x):
    return sigmoid(x)*(1-sigmoid(x))


#神经网络前向传播过程
def forward(x,w1,w2):
     #layer1
    hin1 = x.dot(w1)
    hout1 = sigmoid(hin1)
    #layer2
    # 将h层数据传到o层，并计算输出 将输出的o层激活
    oin = hout1.dot(w2)
    out = sigmoid(oin)

    return out


#定义网络初始参数
'''
N:批量样本数
layer1_in:输入向量维数，样本特征数
layer_out:输出向量维数
'''
if __name__ =="__main__":
    
    starttime = datetime.datetime.now()
#    iris
#    N, layer1_in, layer2_in, layer_out = 10, 4,12,1
#    wine
    N, layer1_in, layer2_in, layer_out = 10, 13,52,1
    #参数初始化，正态分布
    w1 = np.random.normal(size=[layer1_in,layer2_in])
    w2 = np.random.normal(size=[layer2_in,layer_out])
  
    
    #存放损失数据
    losses = []
    #学习速率以 0.01 ~ 0.001 为宜。
    learning_rate = 0.001
    flag = 0
    for step in range(30):
        if flag:
            
            # 计算h层输出,将输出的h层激活
            #layer1
            Houter1=np.zeros(shape=(np.shape(x_train)[0],np.shape(w1)[1]))
            i =0
            for line in x_train:
                hin1 = line.dot(w1)
                hout1 = sigmoid(hin1)
                Houter1[i]=hout1
                i += 1
                          
            # 将h层数据传到o层，并计算输出 将输出的o层激活
            Y_=np.zeros(shape=(np.shape(x_train)[0],np.shape(w2)[1]))
            i =0
            for line in Houter1:
                oin = line.dot(w2)
                out = sigmoid(oin)
                Y_[i]=out
                i += 1
                    
            #扁平化处理
            y_tr=np.reshape(y_train, [np.shape(y_train)[0], -1])
            
#            print(round((np.square(Y_ - y_tr).sum())/N, 6))
            if step % 1 == 0:
                losses.append(round((np.square(Y_ - y_tr).sum())/N, 6)) 
                
            #计算w2的梯度损失
            grad_y_pred = (2 * (Y_ - y_tr))
            grad_o_sig = sigDer(out)
            grad_o = grad_y_pred * grad_o_sig
            grad_w2 = Houter1.T.dot(grad_o)
                    
            #计算w1的梯度损失
            grad_h_relu = grad_y_pred.dot(w2.T)
            grad_h_sig = sigDer(hout1)
            grad_h = grad_h_relu * grad_h_sig
            grad_w1 = x_train.T.dot(grad_h)
            
            
            # 更新梯度
            w1 -= learning_rate * grad_w1
            w2 -= learning_rate * grad_w2
           
        flag = 1
#        #使用训练集进行测试
#        right_n = 0
#        all_n = len(x_train)
#        for i in range(all_n):
#            Y_ = forward(x_train[i],w1,w2)
#            if np.argmax(Y_)==np.argmax(y_train[i]):
#                right_n=right_n+1
#                
#        print("总数---"+str(all_n))
#        print("正确个数"+str(right_n))
#        print("准确率----")
#        print(float(right_n/all_n))
    print(w1,w2)
    #使用测试集进行测试
    right_n = 0
    pre=[]
    all_n = len(x_test)
    for i in range(len(x_test)):
        Y_ = forward(x_test[i],w1,w2)
        pre.append(np.argmax(Y_))
        if np.argmax(Y_)==np.argmax(y_test[i]):
            right_n=right_n+1

    print("总数---"+str(all_n))
    print("正确个数"+str(right_n))
    print("准确率----")
    print(float(right_n/all_n))
    
    
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    print(losses)
    plt.title('wine训练过程中loss变化情况')
    plt.plot(losses)
    plt.show()
#    
    print (classification_report(y_test, pre))
#    endtime = datetime.datetime.now()
#    print ("time",endtime - starttime)