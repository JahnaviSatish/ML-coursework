import pandas as pd
import numpy as np
from numpy.linalg import matrix_rank

df = pd.read_excel("ML-lab2 dataset.xlsx",sheet_name="Purchase data")
'''
print(df)
print(df[['Customer','Candies (#)','Mangoes (Kg)','Milk Packets (#)','Payment (Rs)']])
'''
def feature():
    x=df[['Candies (#)','Mangoes (Kg)','Milk Packets (#)']].to_numpy()
    return x
def output():
    y=df[['Payment (Rs)']].to_numpy()
    return y
def rank(X):
    return matrix_rank(X)
def cost(X,Y):
    c=np.linalg.pinv(X)
    return np.dot(c,Y)
X=feature()
Y=output()
print("feature matrix:",X)
print("output matrix:",Y)
print("rank:",rank(X))
print("cost:",cost(X,Y))

