import math
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_excel("ML-lab2 dataset.xlsx", sheet_name="marketing_campaign")
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"]).astype("int64")

# -----------------------------
# Label Encoding
# -----------------------------

def labelEncoding(df, column):
    data = df.copy()

    values = data[column].dropna().unique()

    labels = {}

    for i, value in enumerate(values):
        labels[value] = i

    data[column] = data[column].map(labels)

    return data, labels


# -----------------------------
# One-Hot Encoding
# -----------------------------

def oneHot(df, column):
    data = df.copy()

    values = data[column].dropna().unique()

    for value in values:
        data[f"{column}_{value}"] = (data[column] == value).astype(int)

    data.drop(column, axis=1, inplace=True)

    return data


# -----------------------------
# Label Encoding
# -----------------------------

encoded_df, labels = labelEncoding(df.copy(), "Education")

print("Encoded Labels")
print(labels)
print(encoded_df.head())

# -----------------------------
# One-Hot Encoding
# -----------------------------

encoded_df = oneHot(encoded_df, "Marital_Status")

print(encoded_df.head())

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


# -----------------------------
# Series
# -----------------------------

def minkowski_series(v1, v2):

    distances = []

    for p in range(1, 11):
        distances.append(minkowski_dist(v1, v2, p))

    return distances


# -----------------------------
# Plot
# -----------------------------

def plot_minkowski(distances):

    p_values = list(range(1, 11))

    plt.plot(p_values, distances, marker="o")
    plt.title("Minkowski Distance")
    plt.xlabel("p")
    plt.ylabel("Distance")
    plt.xticks(p_values)
    plt.grid(True)

    plt.show()


# -----------------------------
# Unit Test
# -----------------------------

print("\nRunning Minkowski Unit Tests...")

v1 = [1, 2, 3, 4]
v2 = [5, 6, 7, 8]

assert manhattan(v1, v2) == 16

assert round(euclidean(v1, v2), 5) == 8

assert minkowski_dist(v1, v1, 3) == 0

print("Minkowski Tests Passed")


print("\nUnit Test Case")

distances = minkowski_series(v1, v2)

for p, d in enumerate(distances, start=1):
    print(f"p = {p}  Distance = {d:.4f}")

plot_minkowski(distances)


# -----------------------------
# Actual Dataset
# -----------------------------

print("\nActual Dataset")

v1 = encoded_df.iloc[0].to_numpy()
v2 = encoded_df.iloc[1].to_numpy()

distances = minkowski_series(v1, v2)

for p, d in enumerate(distances, start=1):
    print(f"p = {p}  Distance = {d:.4f}")

plot_minkowski(distances)