from scipy.spatial.distance import minkowski
import math
# code generation - chatgpt
# -----------------------------
# Euclidean Distance
# -----------------------------

def euclidean(v1, v2):

    assert len(v1) == len(v2)

    distance = 0

    for i in range(len(v1)):
        distance += (v1[i] - v2[i]) ** 2

    return math.sqrt(distance)


# -----------------------------
# Manhattan Distance
# -----------------------------

def manhattan(v1, v2):

    assert len(v1) == len(v2)

    distance = 0

    for i in range(len(v1)):
        distance += abs(v1[i] - v2[i])

    return distance


# -----------------------------
# Minkowski Distance
# -----------------------------

def minkowski_dist(v1, v2, p):

    assert len(v1) == len(v2)

    if p == 1:
        return manhattan(v1, v2)

    elif p == 2:
        return euclidean(v1, v2)

    distance = 0

    for i in range(len(v1)):
        distance += abs(v1[i] - v2[i]) ** p

    return distance ** (1 / p)


v1 = [1, 2, 3, 4]
v2 = [5, 6, 7, 8]

print("Comparison of Custom vs SciPy Minkowski Distance\n")

for p in range(1, 11):
    custom_distance = minkowski_dist(v1, v2, p)
    scipy_distance = minkowski(v1, v2, p)

    print(f"p = {p}")
    print(f"Custom : {custom_distance:.4f}")
    print(f"SciPy  : {scipy_distance:.4f}")
    print()