import pandas as pd
import numpy as np

df = pd.read_excel("ML-lab2 dataset.xlsx", sheet_name="thyroid0387_UCI")
df.replace("?", np.nan, inplace=True)

def binary_attributes():
    binary = []
    for col in df.columns:
        if len(df[col].dropna().unique()) == 2:
            binary.append(col)
    return binary

def observation(binary):
    v1 = df.iloc[0][binary].replace({'t':1, 'f':0, 'M':1, 'F':0})
    v2 = df.iloc[1][binary].replace({'t':1, 'f':0, 'M':1, 'F':0})
    return v1, v2

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

def jaccard(f11, f10, f01):
    return f11 / (f11 + f10 + f01)

def smc(f11, f10, f01, f00):
    return (f11 + f00) / (f11 + f10 + f01 + f00)


# Main Program

binary = binary_attributes()
v1, v2 = observation(binary)

f11, f10, f01, f00 = frequency(v1, v2)
JC = jaccard(f11, f10, f01)
SMC = smc(f11, f10, f01, f00)
print(binary)
print("v1")
print(v1)
print("v2")
print(v2)
print("f11=",f11)
print("f10=",f10)
print("f01=",f01)
print("f00=",f00)
print("Jaccard Coefficient =",JC)
print("Simple Matching Coefficient =",SMC)