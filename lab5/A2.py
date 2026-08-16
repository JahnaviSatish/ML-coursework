import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
df=pd.read_csv("eeg_features.csv")  
X = df.drop(columns=["label","subject"]).copy()
y = df["label"].copy()
#print(X)
#print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)  
#random_state=42  same split every time you run program
# stratify=y --> healthy/schizophrenia class proportions are maintained in both sets.

def missingValues(df):
    return df.isnull().sum()

################ imputation not required here #########################
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


print(missingValues(df))

# to call for imputation, right now not needed
print(y_test)
testvector = X_test.iloc[4].values #3rd value out of 9
dist_rec=all_dist(testvector, X_train.values, y_train.values, "euclidean", p=2)
sorting=sorting_metric("selectionSort",dist_rec)
knn=k_nearest(3,sorting)
pred=weightedknn(knn)
print("distance, class for k value : ",knn)
print("predicted class:",pred)
print("actual class:",y_test.iloc[4])
#print("accuracy:",accuracy(pred,y_test))

#step1- print actual y_test val
#step2-calculate distance array along with the class
#step3- sort the array
#step4- from that find the nearest k neighbors
