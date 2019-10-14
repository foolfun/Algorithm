# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:34:49 2019

@author: zsl
"""
#FP树类定义
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode      #needs to be updated
        self.children = {} 
    
    def inc(self, numOccur):
        self.count += numOccur
        
    def disp(self, ind=1):
        print( '  '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)

#FP树构建函数
def createTree(dataSet, minSup=1): 
    headerTable = {}
    #go over dataSet twice
    for trans in dataSet:#first pass counts frequency of occurance
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    for k in list(headerTable.keys()):  #remove items not meeting minSup
        if headerTable[k] < minSup: 
            del(headerTable[k])
    freqItemSet = set(headerTable.keys())
    #print( 'freqItemSet: ',freqItemSet)
    if len(freqItemSet) == 0: 
        return None, None  #if no items meet min support -->get out
    for k in headerTable:
        headerTable[k] = [headerTable[k], None] #reformat headerTable to use Node link 
    #print('headerTable: ',headerTable)
    retTree = treeNode('Null Set', 1, None) #create tree
    for tranSet, count in dataSet.items():  #go through dataset 2nd time
        localD = {}
        for item in tranSet:  #put transaction items in order
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count)#populate tree with ordered freq itemset
    return retTree, headerTable #return tree and header table

def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:#check if orderedItems[0] in retTree.children
        inTree.children[items[0]].inc(count) #incrament count
    else:   #add items[0] to inTree.children
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None: #update header table 
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:#call updateTree() with remaining ordered items
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)
        
def updateHeader(nodeToTest, targetNode):   #this version does not use recursion
    while (nodeToTest.nodeLink != None):    #Do not use recursion to traverse a linked list!
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode
   
#发现以给定元素项结尾的所有路径的函数     
def ascendTree(leafNode, prefixPath): #ascends from leaf node to root
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)
    
def findPrefixPath(basePat, treeNode): #treeNode comes from header table
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1: 
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

#递归查找频繁项集的mineTree函数
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]#(sort header table)
    for basePat in bigL:  #start from bottom of header table
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        #print 'finalFrequent Item: ',newFreqSet    #append to set
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        #print 'condPattBases :',basePat, condPattBases
        #2. construct cond FP-tree from cond. pattern base
        myCondTree, myHead = createTree(condPattBases, minSup)
        #print 'head from conditional tree: ', myHead
        if myHead != None: #3. mine cond. FP-tree
            #print 'conditional tree for: ',newFreqSet
            #myCondTree.disp(1)            
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)

def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

import twitter
from time import sleep
import re

def textParse(bigString):
    urlsRemoved = re.sub('(http:[/][/]|www.)([a-z]|[A-Z]|[0-9]|[/.]|[~])*', '', bigString)    
    listOfTokens = re.split(r'\W*', urlsRemoved)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def getLotsOfTweets(searchStr):
    CONSUMER_KEY = ''
    CONSUMER_SECRET = ''
    ACCESS_TOKEN_KEY = ''
    ACCESS_TOKEN_SECRET = ''
    api = twitter.Api(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET,
                      access_token_key=ACCESS_TOKEN_KEY, 
                      access_token_secret=ACCESS_TOKEN_SECRET)
    #you can get 1500 results 15 pages * 100 per page
    resultsPages = []
    for i in range(1,15):
        print( "fetching page %d" % i)
        searchResults = api.GetSearch(searchStr, per_page=100, page=i)
        resultsPages.append(searchResults)
        sleep(6)
    return resultsPages

def mineTweets(tweetArr, minSup=5):
    parsedList = []
    for i in range(14):
        for j in range(100):
            parsedList.append(textParse(tweetArr[i][j].text))
    initSet = createInitSet(parsedList)
    myFPtree, myHeaderTab = createTree(initSet, minSup)
    myFreqList = []
    mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
    return myFreqList


from numpy import *
import numpy as np
from sklearn.datasets import load_iris,load_wine
import pandas as pd
#iris
#data=load_iris()
#dataSet = data.data
#labels = data.feature_names
#target = data.target
#r=[]
#for i in target:
#    if i==0:
#        i='Iris-setosa'
#    elif i==1:
#        i='Iris-versicolor'
#    else:
#        i='Iris-virginica'
#    r.append(i)
#simpDat = []
#for i in range(4):
#    tmp=pd.cut(dataSet[:,1],3,labels=['a'+str(i),'b'+str(i),'c'+str(i)])
#    simpDat.append(tmp)
#simpDat.append(r)
#simpDat = np.array(list(zip(*simpDat)))

#wine
#data=load_wine()
#dataSet = data.data
#r = data.target
#simpDat = []
#for i in range(13):
#    tmp=pd.cut(dataSet[:,1],3,labels=['a'+str(i),'b'+str(i),'c'+str(i)])
#    simpDat.append(tmp)
#simpDat.append(r)
#simpDat = np.array(list(zip(*simpDat)))

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split(',')
        # print(lineArr)
        if lineArr[0] == 'vhigh':
            lineArr[0] ='a1'
        if lineArr[0] == 'high':
            lineArr[0] ='a2'
        if lineArr[0] == 'med':
            lineArr[0] ='a3'
        if lineArr[0] == 'low':
            lineArr[0] ='a4'
            
        if lineArr[1] == 'vhigh':
            lineArr[1] ='b1'
        if lineArr[1] == 'high':
            lineArr[1] ='b2'
        if lineArr[1] == 'med':
            lineArr[1] ='b3'
        if lineArr[1] == 'low':
            lineArr[1] ='b4'
            
        if lineArr[2]=='2':
            lineArr[2]='c1'
        if lineArr[2]=='3':
            lineArr[2]='c2'
        if lineArr[2]=='4':
            lineArr[2]='c3'
        if lineArr[2]=='5more':
            lineArr[2]='c4'
            
        if lineArr[3]=='2':
            lineArr[3]='d1'
        if lineArr[3]=='4':
            lineArr[3]='d2'
        if lineArr[3]=='more':
            lineArr[3]='d3'
            
        if lineArr[4]=='small':
            lineArr[4]='e1'
        if lineArr[4]=='med':
            lineArr[4]='e2'
        if lineArr[4]=='big':
            lineArr[4]='e3'
            
        if lineArr[5]=='low':
            lineArr[5]='f1'
        if lineArr[5]=='med':
            lineArr[5]='f2'
        if lineArr[5]=='high':
            lineArr[5]='f3'
      
        dataMat.append(lineArr)
  
    return dataMat
#car
simpDat = loadDataSet('car.data')
    

#initSet = createInitSet(simpDat)
#print(initSet)
#myFPtree, myHeaderTab = createTree(initSet, minSup)
#myFPtree.disp()
#myFreqList = []
#mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
#print(myFreqList)

minSup=500
#simpDat = loadSimpDat()
#print(simpDat)
initSet = createInitSet(simpDat)
#print(initSet)
myFPtree, myHeaderTab = createTree(initSet, minSup)
myFPtree.disp()
myFreqList = []
mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
print(myFreqList)

#parsedDat = [line.split() for line in open('kosarak.dat').readlines()]
#initSet = createInitSet(parsedDat)
#myFPtree, myHeaderTab = createTree(initSet, 100000)
#myFPtree.disp()
#myFreqList = []
#mineTree(myFPtree, myHeaderTab, 100000, set([]), myFreqList)
#print(myFreqList)
