import math
import numpy as np
import pandas as pd
df = pd.read_excel('ML-lab2 dataset.xlsx', sheet_name='marketing_campaign')

def mean(x):
    total = 0

    for value in x:
        total += value

    return total / len(x)


def variance(x):
    avg = mean(x)
    var = 0

    for value in x:
        var += (value - avg) ** 2

    return var / len(x)


def sd(x):
    return math.sqrt(variance(x))

# Sample Data
x = [2, 4, 4, 4, 5, 5, 7, 9]

print("Mean :", mean(x))
print("Variance :", variance(x))
print("Standard Deviation :", sd(x))
# using assert
assert mean(x) == 5.0
assert variance(x) == 4.0
assert sd(x) == 2.0

print("Mean, Variance and Standard Deviation Test: Passed")
# actual df

feature = df["Income"].dropna().tolist()
print("feature: income")
print("Mean :", mean(feature))
print("Variance :", variance(feature))
print("Standard Deviation :", sd(feature))
mean_numpy = np.mean(feature)
std_numpy = np.std(feature)

print("Mean(using numpy):", mean_numpy)
print("sd(using numpy):", std_numpy)