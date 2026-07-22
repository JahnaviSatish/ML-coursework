import pandas as pd
import numpy as np
from scipy.spatial import distance
import random
def minkowski_dist(v1,v2,p):
    dist=0
    for i,j in zip(v1,v2):
        dist+=(abs(i-j))**p
    dist=dist/p
    return dist
#def distance_classification(p)
A=[1,0,0]
B=[2,4,0]
p=random.randint(1,3)
minkowski=minkowski_dist(A,B,p)
print("minkowski dist=",minkowski)
print(p)
