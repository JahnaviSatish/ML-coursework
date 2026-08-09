# logic was given, code module by chatgpt
import math
import time
import tracemalloc
def euclidean(v1, v2):
    distance = 0

    for i in range(len(v1)):
        distance += (v1[i] - v2[i]) ** 2

    return math.sqrt(distance)


def manhattan(v1, v2):
    distance = 0

    for i in range(len(v1)):
        distance += abs(v1[i] - v2[i])

    return distance


def minkowski_dist(v1, v2, p):

    if p == 1:
        return manhattan(v1, v2)

    elif p == 2:
        return euclidean(v1, v2)

    distance = 0

    for i in range(len(v1)):
        distance += abs(v1[i] - v2[i]) ** p

    return distance ** (1 / p)

def performance_test(func, *args):

    # Start measuring memory
    tracemalloc.start()

    # Start timer
    start = time.perf_counter()

    # Execute the function
    result = func(*args)

    # Stop timer
    end = time.perf_counter()

    # Get memory usage
    current, peak = tracemalloc.get_traced_memory()

    # Stop memory tracking
    tracemalloc.stop()

    print("Execution Time:", end - start, "seconds")
    print("Peak Memory:", peak / 1024, "KB")

    return result
# unit test case
v1 = [1, 2, 3, 4]
v2 = [5, 6, 7, 8]

print("Euclidean Distance :", euclidean(v1, v2))
print("Manhattan Distance :", manhattan(v1, v2))
print("Minkowski Distance (p=3) :", minkowski_dist(v1, v2, 3)) # generalised
print("Minkowski Distance (p=1) :", minkowski_dist(v1, v2, 1)) # manhattan always order 1
print("Minkowski Distance (p=2) :", minkowski_dist(v1, v2, 2)) # eculidean always order 2
print("performance for manhattan")
print(performance_test(manhattan,v1,v2))
print("performance for euclidean")
print(performance_test(euclidean,v1,v2))
print("performance for minkowski")
print(performance_test(minkowski_dist,v1,v2,3))

