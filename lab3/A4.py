import numpy as np
def minkowski_dist(v1,v2,p):
    dist=0
    for i,j in zip(v1,v2):
        dist+=(abs(i-j))**p
    dist=dist**(1/p)
    return dist
def euclidean(v1,v2):
    euclidean=0
    for i,j in zip(v1,v2):
        euclidean+=(j-i)**2
    euclidean=np.sqrt(euclidean)
    return euclidean
def manhattan(v1,v2):
    manhattan=0
    for i,j in zip(v1,v2):
        manhattan+=abs(i-j)
    return manhattan
# note to self- taken from w3schools
# The zip() function returns a zip object, which is an iterator of tuples 
# where the first item in each passed iterator is paired together, 
# and then the second item in each passed iterator are paired together etc.
# If the passed iterables have different lengths, the iterable with the least items 
# decides the length of the new iterator.
A=[1,0,0]
B=[2,4,0]
p = int(input("Enter order p>0: "))
minkowski=minkowski_dist(A,B,p)        
print("minkowski distance=",minkowski)
print("order value=",p)
if p==1:# manhatten is always order 1
    m=manhattan(A,B)
    print("manhatten distance=",m)
elif p==2:#euclidean order 2
    e=euclidean(A,B)
    print("euclidean distance=",e)
else:
    print("neither manhattan nor euclidean as p is greater than order 2")
