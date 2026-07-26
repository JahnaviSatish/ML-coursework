
import pandas as pd
import numpy as np
df=pd.read_excel('ML-lab2 dataset.xlsx',sheet_name=('marketing_campaign'))
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"]).astype("int64")# making date as an integer or have to drop column
df = df.select_dtypes(exclude="object")
# correct output would be to encode the categoric columns from dataset
# but since it wasnt mentioned to use dataset for this question
# i am taking 2 vectors from dataset but dropping all the categorical data points
def dotproduct(v1, v2):
    dot = 0
    for i, j in zip(v1, v2):
        dot += i * j# notes on zip function there in A4
    return dot

def e_norm(v): #formula= sqrt(sum of all vector points^2)
    sum=0
    for i in v:
        sum+=i**2
    return np.sqrt(sum)

A=df.iloc[0].to_numpy()
B=df.iloc[1].to_numpy()
mydot = dotproduct(A, B)
npdot = np.dot(A, B)
ownNormA = e_norm(A)
npNormA = np.linalg.norm(A)
ownNormB= e_norm(B)
npNormB = np.linalg.norm(B)
print("Own Dot Product :", mydot)
print("NumPy Dot Product :", npdot)
print("Own Norm of A :", ownNormA)
print("NumPy Norm of A :", npNormA)
print("Own Norm of B :", ownNormB)
print("NumPy Norm of B :", npNormB)