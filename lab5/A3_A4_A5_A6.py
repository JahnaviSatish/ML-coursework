import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
df=pd.read_csv("eeg_features.csv")  
X = df.drop(columns=["label","subject"]).copy()
y = df["label"].copy()
print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)  
#random_state=42  same split every time you run program
# stratify=y --> healthy/schizophrenia class proportions are maintained in both sets.
 #### A4 ######
neigh = KNeighborsClassifier(n_neighbors=3)  
neigh.fit(X_train, y_train)  
# fit finds the nearest k neighbors in the training set
####### A5 #######
accuracy=neigh.score(X_test, y_test)  
####### A6 ########
prediction=neigh.predict(X_test) 
print("checking how data has been split:")
print(X.shape,y.shape)
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)
print("knn")
print("accuracy=",accuracy)
print("prediction=",prediction)