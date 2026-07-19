# reusing functions from A4 and A5
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel("ML-lab2 dataset.xlsx", sheet_name="thyroid0387_UCI")
df1 = df.copy() #making another df to encode for cosine similarity
df.replace("?", np.nan, inplace=True)
data = df.iloc[:20]

for col in df1.select_dtypes(include='object').columns:
    df1[col] = pd.factorize(df1[col])[0]  #converting obj dtype to numeric


def binary_attributes(): # collecting only binary value columns for JC and SMc
    binary = []
    for col in df.columns:
        if len(df[col].dropna().unique()) == 2:
            binary.append(col)
    return binary

def observation(binary):
    binary_data = data[binary].replace({'t':1,'f':0,'M':1,'F':0}) #instead of doing conversion one by one, doing it for all 20 rows
    return binary_data

def frequency(v1, v2):
    f11 = 0
    f10 = 0
    f01 = 0
    f00 = 0

    for a, b in zip(v1, v2):

        if a == 1 and b == 1:
            f11 += 1

        elif a == 1 and b == 0:
            f10 += 1

        elif a == 0 and b == 1:
            f01 += 1

        else:
            f00 += 1

    return f11, f10, f01, f00


def jaccard(f11, f10, f01):# same as before
    return f11 / (f11 + f10 + f01)


def smc(f11, f10, f01, f00):#same as before
    return (f11 + f00) / (f11 + f10 + f01 + f00)


def dotp(v1, v2): # dot product
    return np.dot(v1, v2)

def magnitude(v):
    return np.sqrt(np.sum(v ** 2))

def cosine(v1, v2):
    return dotp(v1, v2) / (magnitude(v1) * magnitude(v2))


binary = binary_attributes()
binary_data = observation(binary)
complete = df1.iloc[:20] #for cosine similarity

JC = np.zeros((20,20)) # empty matrices
SMC = np.zeros((20,20)) # ready to be filled
COS = np.zeros((20,20))

for i in range(20):
    for j in range(20): #accessing each pair using matrix operation
        v1 = binary_data.iloc[i]
        v2 = binary_data.iloc[j]
        #using original df for jc and smc
        f11, f10, f01, f00 = frequency(v1, v2)
        JC[i][j] = jaccard(f11, f10, f01)
        SMC[i][j] = smc(f11, f10, f01, f00)

        #using df1 for cosine similarity
        c1 = complete.iloc[i]
        c2 = complete.iloc[j]

        COS[i][j] = cosine(c1, c2)


plt.figure(figsize=(8,6))#fixing size for heatmap
sns.heatmap(JC, annot=True)
plt.title("jaccard coefficient")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(SMC, annot=True)
plt.title("simple matching coefficient")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(COS, annot=True)
plt.title("cosine cimilarity")
plt.show()