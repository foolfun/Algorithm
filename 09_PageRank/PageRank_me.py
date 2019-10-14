# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 18:44:07 2019

@author: zsl
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


f = open('input.txt', 'r')
edges = [line.strip('\n').split(' ') for line in f]
 
G = nx.DiGraph()
for edge in edges:
    G.add_edge(edge[0], edge[1])
nx.draw(G, with_labels=True)
plt.show()

filename ='input.txt'
def load_data(f):
    data = open(f,'r')
    edges = [line.strip('\n').split(' ') for line in data]
    return edges

def get_Value(edges):
    nodes = []
    for edge in edges:
        for i in edge:
            if i not in nodes:
                nodes.append(i)
    print(nodes)
    
    N=len(nodes)
    i = 0
    node_to_num = {}
    for node in nodes:
        node_to_num[node] = i
        i += 1
    for edge in edges:
        edge[0] = node_to_num[edge[0]]
        edge[1] = node_to_num[edge[1]]
    print(edges)
    
    M = np.zeros([N, N])
    for edge in edges:
        M[edge[1], edge[0]] = 1
    print(M)
    for j in range(N):
        sum_cnt = sum(M[:,j])
        for k in range(N):
            if sum_cnt !=0:
                M[k,j]/=sum_cnt
            else:
                M[k,j]=1/N
    print('=================',M)
    V = np.ones(N)/N
    return M,V,N
        
def PageRank(M,V,N,beta=0.85):
    cnt =0
    e=np.ones([N,N])/N
    M_ = np.dot(M,beta)+np.dot(1-beta,e)
    er =100000
    pr1=V
    re=[]
    while er > 0.00000001:
        pr2 = np.dot(M_,pr1)
        er =pr2-pr1
        er = max(map(abs, er))
        pr1 = pr2
        cnt +=1
        print('iteration %s'%str(cnt),pr1)         
        if cnt%10==0:
            re.append(pr1.tolist())
    return pr1,re

data=load_data(filename)
M,V,N = get_Value(data)
pr1,re = PageRank(M,V,N)
print(re)
re = np.array(re)
ln1, = plt.plot(re[:,0], color = 'red', linewidth = 2.0, linestyle = '--')
ln2, = plt.plot(re[:,1], color = 'blue', linewidth = 2.0, linestyle = '--')
ln3, = plt.plot(re[:,2], color = 'pink', linewidth = 2.0, linestyle = '--')
ln4, = plt.plot(re[:,3], color = 'green', linewidth = 2.0, linestyle = '--')
ln5, = plt.plot(re[:,4], color = 'yellow', linewidth = 2.0, linestyle = '--')
plt.legend(handles=[ln1,ln2,ln3,ln4,ln5], labels=['A', 'B','C','D','F'],
    loc='lower right')
plt.plot(re)

plt.bar(['A', 'B','C','D','F'], pr1)
plt.show()

