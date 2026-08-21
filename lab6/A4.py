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
#### for this dataset imputation is not required as it is already cleaned
#### but for lab i am writing imputation function
def missingValues(df):
    return df.isnull().sum()
'''
def outliers(X):
    outliercols = []
    for col in numdf:
        Q1 = numdf[col].quantile(0.25)
        Q3 = numdf[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = numdf[(numdf[col] < lower) | (numdf[col] > upper)]
        if len(outliers) > 0:
            outliercols.append(col)
    return numdf,outliercols

def imputation(): # no mode coz no categorical data
    numdf,outlierColumns = outliers()
    for col in numdf:
        if col in outlierColumns:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mean())

'''
################ imputation done #########################
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

def accuracy(preds, y_test):
    return (preds == y_test).mean()


#print(missingValues(df))
numdf=df.select_dtypes(include=np.number)
'''
print(outliers(X))
print(X)
''' 

#step1- print actual y_test val
#step2-calculate distance array along with the class
#step3- sort the array
#step4- from that find the nearest k neighbors


results = []

for k in range(1, 10):

    correct = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(len(X_test)):

        testvector = X_test.iloc[i].values

        dist_rec = all_dist(
            testvector,
            X_train.values,
            y_train.values,
            "euclidean",
            p=2
        )

        sorting = sorting_metric("selectionSort", dist_rec)

        knn = k_nearest(k, sorting)

        pred = tie_breaker(knn)

        actual = y_test.iloc[i]

        if pred == actual:
            correct += 1

        if actual == 1 and pred == 1:
            TP += 1

        elif actual == 0 and pred == 0:
            TN += 1

        elif actual == 0 and pred == 1:
            FP += 1

        elif actual == 1 and pred == 0:
            FN += 1

    # =========================
    # Calculate Metrics
    # =========================

    total = TP + TN + FP + FN

    accuracy_value = (TP + TN) / total

    if TP + FP != 0:
        precision = TP / (TP + FP)
    else:
        precision = 0

    if TP + FN != 0:
        recall = TP / (TP + FN)
    else:
        recall = 0

    if precision + recall != 0:
        f1Score = (
            2 * precision * recall
        ) / (precision + recall)
    else:
        f1Score = 0

    results.append(
        (
            k,
            accuracy_value,
            precision,
            recall,
            f1Score
        )
    )


# =========================
# Display Results
# =========================

print("\nKNN Performance for Different K Values")
print("-" * 70)

print(
    f"{'K':<5}"
    f"{'Accuracy':<15}"
    f"{'Precision':<15}"
    f"{'Recall':<15}"
    f"{'F1 Score':<15}"
)

print("-" * 70)

for k, accuracy_value, precision, recall, f1Score in results:

    print(
        f"{k:<5}"
        f"{accuracy_value:<15.4f}"
        f"{precision:<15.4f}"
        f"{recall:<15.4f}"
        f"{f1Score:<15.4f}"
    )