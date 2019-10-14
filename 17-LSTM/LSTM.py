# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:09:43 2019

@author: zsl
"""
from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import torch
import torch.nn as nn

#读取数据
def findFiles(path): return glob.glob(path)

#print(findFiles('data/names/*.txt'))

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'#[Mn] Mark, Nonspacing
        and c in all_letters
    )


# 获取所有种类的语言以及每个语言包含的所有姓氏
all_categories = []
category_lines = {}

# 划分训练集和测试集
training_lines = {}
validation_lines = {}


def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines
    
    num_of_training_set = int(len(lines)*0.8)
    training_lines[category]   = lines[:num_of_training_set]
    validation_lines[category] = lines[num_of_training_set:] 

#print(all_categories)
    
    
#把每个词转换为词向量
# 把每个字母转换为在字母表中的位置
def letterToIndex(letter):
    return all_letters.find(letter)

# 将字母对应的数字转换为onehot编码
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

# 将一个单词转换为若干字母向量的组合
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

#
print(letterToTensor('J'))

print(lineToTensor('Jones').size())
#掉包法：
#class BaseLSTM(torch.nn.Module):
#    def __init__(self,input_size, hidden_size, num_layers):
#        super().__init__()
#        self.rnn=torch.nn.LSTM(
#            input_size=input_size,
#            hidden_size=hidden_size,
#            num_layers=1,
#            batch_first=True
#        )
#        self.out=torch.nn.Linear(in_features=hidden_size,out_features=18)
#
#    def forward(self,x):
##        print(x.size())
#        output,(h_n,c_n)=self.rnn(x)
##        print(output.si#ze())
#        output_in_last_timestep=output[:,-1,:] # 也是可以的
##        output_in_last_timestep=h_n[-1,:,:]
##        print(output_in_last_timestep.size())
##        print(output_in_last_timestep.equal(output[:,-1,:])) #ture
#        x=self.out(output_in_last_timestep)
##        print(x.size())
#        return x
#
#n_hidden = 128
#n_categories = len(all_categories)
#rnn = BaseLSTM(n_letters, n_hidden, n_categories)
#if torch.cuda.is_available():
#    rnn = rnn.cuda()
#自己搭建的：
class BaseLSTM(nn.Module):
    def __init__(self, input_size,hidden_size,output_size):
        super(BaseLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # 定义weight维度
        self.wf = nn.Linear(input_size,  hidden_size)
        self.wi = nn.Linear(input_size,  hidden_size)
        self.wz = nn.Linear(input_size,  hidden_size)
        self.wo = nn.Linear(input_size,  hidden_size)
        self.wy = nn.Linear(hidden_size,  output_size)

        self.activation = nn.Tanh()
        self.activation0 = nn.Sigmoid()
        
    def step(self, letter, hidden,c):
        
        zf = self.activation0(self.wf( letter )+self.wf( hidden ))
        zi = self.activation0(self.wi( letter )+self.wi( hidden ))
        z  = self.activation0(self.wz( letter )+self.wi( hidden ))
        zo = self.activation0(self.wo( letter )+self.wi( hidden ))
        
        c = zf * c + zi*z
        
        hidden = zo * self.activation(c)
#        output = self.activation0(self.wy(hidden))#或者
        output = self.wy(hidden)
        return output, hidden

    def initHidden(self, is_cuda=True):
        if torch.cuda.is_available():
            return torch.zeros(1, self.input_size).cuda()
        else:
            return torch.zeros(1, self.input_size)

    def forward(self, word):
        hidden = self.initHidden()
        c = self.initHidden()
        for i in range(word.size()[0]):
            output, hidden = self.step(word[i], hidden,c)
        return output

n_hidden = n_letters
n_categories = len(all_categories)
rnn = BaseLSTM(n_letters, n_hidden, n_categories)
if torch.cuda.is_available():
    rnn = rnn.cuda()
    
class DeeperRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeeperRNN, self).__init__()
        self.input1_size = input_size
        self.input2_size = input_size
        self.layer1 = BaseLSTM(input_size, hidden_size, output_size)
        self.layer2 = BaseLSTM(hidden_size, hidden_size, output_size)
        
    def step(self, letter, hidden1, hidden2,c1,c2):
        output1, hidden1 = self.layer1.step(letter, hidden1,c1)
        output2, hidden2 = self.layer2.step(hidden1, hidden2,c2)
        return output2, hidden1, hidden2
    
    def forward(self, word):
        hidden1, hidden2 = self.initHidden()
        c1, c2 = self.initHidden()
        for i in range(word.size()[0]):
            # Only the last output will be used to predict (many to one)
            output, hidden1, hidden2 = self.step(word[i], hidden1, hidden2,c1,c2)
        return output
        
    def initHidden(self, is_cuda=True):
        if is_cuda:
            return torch.zeros(1, self.input1_size).cuda(), torch.zeros(1, self.input2_size).cuda()
        else:
            return torch.zeros(1, self.input1_size), torch.zeros(1, self.input2_size)

n_hidden = n_letters
n_categories = len(all_categories)
rnn = DeeperRNN(n_letters, n_hidden, n_categories)
rnn = rnn.cuda() if torch.cuda.is_available() else rnn

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i
#
#print(categoryFromOutput(output))
    

#训练模型
#获取预测结果
#定义随机选择的函数作为训练集输入 
import random

def randomChoice(l):
    #  random.choice(l)更加简单
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    #随机选择语言
    category = randomChoice(all_categories)
    #随机选择语言对应的某个单词
    line = randomChoice(training_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    category_tensor = category_tensor.cuda() if torch.cuda.is_available() else category_tensor
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

def randomValidationExample():
    category = randomChoice(all_categories)
    line = randomChoice(validation_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    category_tensor = category_tensor.cuda() if torch.cuda.is_available() else category_tensor
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)

#训练函数 
criterion = nn.CrossEntropyLoss()

learning_rate = 0.005

def train(category_tensor, line_tensor):
    output = rnn(line_tensor)
    output = output[torch.arange(output.size(0))==output.size(0)-1]
    rnn.zero_grad()
    loss = criterion(output, category_tensor)
#    print("output======",output)
    loss.backward()
    # 没有优化器的step，所以需要手动更新权值
    for p in rnn.parameters():
        if hasattr(p.grad, "data"):
            p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()

#训练过程
import time
import math

n_iters = 5000
print_every = 500
plot_every = 100

current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
#绘制损失函数
import matplotlib.pyplot as plt
plt.plot(list(range(len(all_losses))), all_losses)
plt.xlabel('iterator')
plt.ylabel('loss values')
plt.show()

#使用混淆矩阵来评估模型
confusion_training   = torch.zeros(n_categories, n_categories)
confusion_validation = torch.zeros(n_categories, n_categories)
n_confusion = 500
def evaluate(line_tensor):
    rnn.eval()
    output = rnn(line_tensor)
    return output

# 训练集混淆矩阵
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion_training[category_i][guess_i] += 1

    
# 验证集混淆矩阵
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomValidationExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion_validation[category_i][guess_i] += 1
    
    
# 按对角线求和的方法计算准确率
right_train = 0
right_valid = 0
for i in range(n_categories):
    right_train += confusion_training[i][i]
    right_valid += confusion_validation[i][i]
acc_train = right_train / n_confusion
acc_valid = right_valid / n_confusion

# 每一行除以该行之和来归一化
for i in range(n_categories):
    confusion_training[i] = confusion_training[i] / confusion_training[i].sum()
    confusion_validation[i] = confusion_validation[i] / confusion_validation[i].sum()


# 画图
fig = plt.figure()
ax1 = fig.add_subplot(121)
cax1 = ax1.matshow(confusion_training.numpy())

ax2 = fig.add_subplot(122)
cax2 = ax2.matshow(confusion_validation.numpy())


ax1.set_xticklabels([''] + all_categories, rotation=90)
ax1.set_yticklabels([''] + all_categories)
ax2.set_xticklabels([''] + all_categories, rotation=90)

plt.show()

print("Traing set Acc is", acc_train.item())
print("validation set Acc is", acc_valid.item())

#预测输入的名字来自于哪种语言，并给出最高的三种可能
def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))
        output = torch.nn.functional.softmax(output, dim=1)

        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('Probability (%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

predict('Dovesky')
predict('Jackson')
predict('Satoshi')
predict("Cui")
predict("Zhuang")
predict("Xue")
predict("Wang")
