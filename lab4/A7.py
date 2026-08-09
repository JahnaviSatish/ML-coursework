import math
import numpy as np
# logic given, code by chatgpt
def dotproduct(v1, v2):
    dot = 0

    for i in range(len(v1)):
        dot += v1[i] * v2[i]

    return dot


def e_norm(v):
    norm = 0

    for i in range(len(v)):
        norm += v[i] ** 2

    return math.sqrt(norm)

v1 = [1, 2, 3, 4]
v2 = [5, 6, 7, 8]
npdot = np.dot(v1, v2)
npNormA = np.linalg.norm(v1)
npNormB = np.linalg.norm(v2)
print("Dot Product :", dotproduct(v1, v2))
print("Euclidean Norm of v1 :", e_norm(v1))
print("Euclidean Norm of v2 :", e_norm(v2))
print("Dot Product using linalg :", npdot)
print("Euclidean Norm of v1 using linalg :", npNormA)
print("Euclidean Norm of v2 using linalg :", npNormB)

