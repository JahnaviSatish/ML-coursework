import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
df=pd.read_csv("eeg_features.csv")  
X = df.drop(columns=["label","subject"]).copy()
y = df["label"].copy()
#print(X)
#print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)  
#random_state=42  same split every time you run program
# stratify=y --> healthy/schizophrenia class proportions are maintained in both sets.
#### for this dataset imputation is not required as it is already cleaned
#### but for lab i am writing imputation function
def missingValues(df):
    return df.isnull().sum()

################ imputation not needed here #########################
def euclidean(v1, v2):
    dist = 0
    for i in range(len(v1)):
        dist += (v1[i] - v2[i]) ** 2
    return math.sqrt(dist)

def manhattan(v1, v2):
    dist = 0
    for i in range(len(v1)):
        dist += abs(v1[i] - v2[i])
    return dist

def minkowski_dist(v1, v2, p):
    dist = 0
    for i in range(len(v1)):
        dist += abs(v1[i] - v2[i]) ** p
    dist=dist ** (1 / p)
    return dist

# calculating vector for the n-dimensions;
def all_dist(testvector, X_train, y_train, metric, p=2):
    distances=[]
    for trainvec, j in zip(X_train,y_train):
        if metric == "euclidean":
            dist= euclidean(trainvec, testvector)
        elif metric == "manhattan":
            dist= manhattan(trainvec, testvector)
        elif metric == "minkowski":
            dist= minkowski_dist(trainvec, testvector, p )
        distances.append((dist,j))
    return distances
# now got a distance array for all vectors 
########## distance metric done ###############
def bubbleSort(arr):
    for i in range(len(arr)):
        for j in range(0, len(arr)-i-1):
            if arr[j]>arr[j + 1]:
                temp=arr[j]
                arr[j]=arr[j+1]
                arr[j+1]=temp
    return arr

def selectionSort(arr, size):
    for ind in range(size-1):
        min_index=ind
        for j in range(ind+1, size):
            if arr[j]<arr[min_index]:
                min_index=j

        arr[ind], arr[min_index] = arr[min_index], arr[ind]
    return arr

def insertionSort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i-1
        # Compare key with each element on the left of it until an element smaller than it is found       
        while j >= 0 and key < arr[j]:
            arr[j+1] = arr[j]
            j =j-1
        arr[j+1] = key
    return arr

def sorting_metric(metric,dist_rec):
    if metric=="selectionSort":
        return selectionSort(dist_rec, len(dist_rec))
    elif metric=="insertionSort":
        return insertionSort(dist_rec)
    elif metric=="bubbleSort":
        return bubbleSort(dist_rec)
    else :
        return 0
############# sorting modules ##############
def k_nearest(k,arr):
    k_arr=[]
    for i in range(k):
        x=arr[i]
        k_arr.append(x)
    return k_arr 
# k nearest neighbors based on distance

def tie_breaker(knn):
    # 1- if same number of votes for a class if k is even
    class0= 0
    class1= 0
    class0_dist= 0
    class1_dist= 0
    for dist, label in knn:
        if label == 0:
            class0+=1
            class0_dist+=dist
        else:
            class1+=1
            class1_dist+=dist
    if class0>class1:
        return 0
    elif class1>class0:
        return 1
    else:
        # choose class whose neighbors have smaller total distance
        if class0_dist < class1_dist:
            return 0
        else:
            return 1


def weightedknn(knn):
    weight0= 0
    weight1= 0
    for dist, label in knn:
        weight=1/dist
        if label == 0:
            weight0+=weight   
        else:
            weight1+=weight
        
    if weight0>weight1:
        return 0
    else:
        return 1


def accuracy(preds, y_test):
    return (preds == y_test).mean()


#print(missingValues(df))

print(y_test)
K=[1,2,3,4,5,6,7,8]


print("training samples:", len(X_train))
print("test samples:", len(X_test))
print("actual test labels:")
print(y_test.values)
normal_acc=[]
weighted_acc=[]
sklearn_acc=[]
for k in K:
    print("k=",k)
    normalpred=[]
    weightedpred=[]
    for testvector in X_test.values:

        dist_rec=all_dist(testvector, X_train.values, y_train.values, "euclidean", p=2)
        sorting=sorting_metric("selectionSort",dist_rec)
        knn=k_nearest(k,sorting)
        pred=tie_breaker(knn)
        normalpred.append(pred)

    print("predicted class own knn:",normalpred)
    na=accuracy(normalpred,y_test.values)
    normal_acc.append(na)
    print("accuracy:own knn",na)

    for testvector in X_test.values:

        dist_rec=all_dist(testvector, X_train.values, y_train.values, "euclidean", p=2)
        sorting=sorting_metric("selectionSort",dist_rec)
        knn=k_nearest(k,sorting)
        pred=weightedknn(knn)
        weightedpred.append(pred)

    print("predicted class weighted knn:",weightedpred)
    wa=accuracy(weightedpred,y_test.values)
    weighted_acc.append(wa)
    print("accuracy weighted knn:",wa)

    sklearn_knn = KNeighborsClassifier( n_neighbors=k)
    sklearn_knn.fit(X_train,y_train)

    sklearn_accuracy = sklearn_knn.score( X_test,y_test)
    print("Sklearn Normal KNN accuracy  :",sklearn_accuracy )
    sklearn_acc.append(sklearn_accuracy)
    print()



plt.plot(K, normal_acc, marker='o', label='My KNN')
plt.plot(K, weighted_acc, marker='o', label='My Weighted KNN')
plt.plot(K, sklearn_acc, marker='o', label='Sklearn KNN')
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs K")
plt.xticks(K)
plt.legend()
plt.grid(True)

plt.show()
