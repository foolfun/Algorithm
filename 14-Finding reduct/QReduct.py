import numpy as np
from itertools import combinations

 
# Convert string column to integer
def str_column_to_int(dataset, column):        
        for row in dataset:
                row[column] = int(row[column])


def dependency(dataset,num):
        
        total=0
        dependency=0
        for j in range(3):        
                fold=list()
                for i in range(len(dataset)):                        
                        data=dataset[i]
                        #print(i)
                        if data[-1]==j:
                                fold.append(dataset[i])
                #print("Fold {}".format(fold))
                count=len(fold)
                #print("count {}".format(count))
                for k in range(len(fold)):
                        list1=fold[k]                
                        for l in range(len(dataset)):
                                #print("len{}".format(len(fold)))
                                list2=dataset[l]
                                if list1[:num]==list2[:num] and list1[-1]!=list2[-1]:                                        
                                        count = count-1
                                        #print("Count inside {}".format(count))
                                        break
                total=total+count
                print("total {}",format(total))
        dependency=total/len(dataset)
        print("{}".format(dependency))

        return dependency

def generate_new_dataset(row,l):
        #print(row)       
        X = np.empty((1628, 0))
        #print(X)
        for i in range(len(row)):
                col=row[i]
                x=[row_new[col] for row_new in dataset]
                x=np.array([x])
                #print(np.transpose(x))
                #print(x)
                x=x.T
                X=np.append(X, x, axis=1)
      
        X=np.array(X).tolist()
        #print("X {}".format(X))
        return X

#car
def loadDataSet(fileName):
    dataMat = []
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
        
        if lineArr[6] == 'unacc':
            lineArr[6] =0
        elif lineArr[6] == 'acc':
            lineArr[6] =1
        elif lineArr[6] == 'good':
            lineArr[6] =2
        else:
            lineArr[6] =3
        dataMat.append([float(lineArr[0]),float(lineArr[1]), float(lineArr[2]),
                        float(lineArr[3]),float(lineArr[4]),
                        float(lineArr[5]),float(lineArr[6])])
       

    return dataMat
      
filename = 'car.data'
dataset = loadDataSet(filename)

# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
#print(dataset)
#this is fuzzify input based on class belongin granulation
dp=dependency(dataset,4)

n=4
initial_val=[0,1,2,3]
comb=combinations([0,1,2,3],3)

co=[i for i in combinations([0,1,2,3],3)]
c=len(co)
tmpde=[]
while n>1 :
        
        for row in  comb:
                
                data_X=generate_new_dataset(row,len(row))
                #print(data_X)
                dp_data_X=dependency(data_X,len(row))
                tmpde.append(dp_data_X)
                if dp > dp_data_X:
                        c=c-1                        
                        continue                
                else:
                        n=n-1
                        prev_row=row
                        
                        break
        if c == 0:
                break
        comb=combinations(row,n)
        comb_temp=[i for i in combinations(row,n)]
        c=len(comb_temp)
        
                
print("final reduct {}".format(prev_row))
        


        
        

        
