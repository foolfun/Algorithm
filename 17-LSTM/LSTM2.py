# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 00:00:17 2019

@author: zsl
"""

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM, CuDNNLSTM
from keras.datasets import imdb
from keras.layers.core import Dropout
import torch
import torch.nn as nn


max_features = 20000  #
# cut texts after this number of words (among top max_features most common words)
maxlen = 80  # 一句话中最大长度为80
batch_size = 32  # 一次输入32句

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)  #  load_data方法中58行修改过，应该是这个 with np.load(path) as f:，没有allow_pickle=True
print((x_train, y_train), (x_test, y_test))
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

# print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
print(x_train)
print('x_train shape:', x_train.shape)
# x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
# print('x_test shape:', x_test.shape)
# exit()

#掉包法：
class BaseLSTM(torch.nn.Module):
    def __init__(self,input_size, hidden_size):
        super().__init__()
        self.rnn=torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.out=torch.nn.Linear(in_features=hidden_size,out_features=18)

    def forward(self,x):
#        print(x.size())
        output,(h_n,c_n)=self.rnn(x)
        output_in_last_timestep=output[:,-1,:]
        x=self.out(output_in_last_timestep)
#        print(x.size())
        return x

n_hidden = 128

rnn = BaseLSTM(n_letters, n_hidden, n_categories)
if torch.cuda.is_available():
    rnn = rnn.cuda()