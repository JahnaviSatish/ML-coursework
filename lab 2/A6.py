#note to self-cosine similarity requires all values to be numeric, so first need to encode the categorical attributes.
import pandas as pd
import numpy as np

df = pd.read_excel("ML-lab2 dataset.xlsx", sheet_name="thyroid0387_UCI")
df1 = df.copy()

for col in df1.select_dtypes(include='object').columns:
    df1[col] = pd.factorize(df1[col])[0]  #converting obj dtype to numeric

v1 = df1.iloc[0]#observation vec taken from the df where everything is converted to numeric
v2 = df1.iloc[1]

def dotp(v1, v2):# dot product
    return np.dot(v1, v2)

def magnitude(v):
    return np.sqrt(np.sum(v ** 2))

def cosine(v1, v2):
    return dotp(v1, v2) / (magnitude(v1) * magnitude(v2))

dp=dotp(v1,v2)
mag1=magnitude(v1)
mag2=magnitude(v2)
cos = cosine(v1, v2)
print("Cosine Similarity =", cos)

# Cosine Similarity Formula: for my reference
# Cos(A, B) = (A · B) / (||A|| × ||B||)
# where,
# A · B   = Dot product of vectors A and B
# ||A||   = square root of (square each of its components then add them together) for vector A
# ||B||   = square root of (square each of its components then add them together) for vector B
# magnitude is also = length of a vector