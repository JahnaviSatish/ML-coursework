import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split

# =========================
# Load Dataset
# =========================

df = pd.read_csv("eeg_features.csv")

X = df.drop(columns=["label", "subject"]).copy()
y = df["label"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Convert DataFrames/Series to NumPy arrays
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


# =========================
# Distance Functions
# =========================

def euclideanDistance(v1, v2):
    distance = 0

    for i in range(len(v1)):
        distance += (v1[i] - v2[i]) ** 2

    return math.sqrt(distance)


def manhattanDistance(v1, v2):
    distance = 0

    for i in range(len(v1)):
        distance += abs(v1[i] - v2[i])

    return distance


def minkowskiDistance(v1, v2, p):

    if p == 1:
        return manhattanDistance(v1, v2)

    elif p == 2:
        return euclideanDistance(v1, v2)

    distance = 0

    for i in range(len(v1)):
        distance += abs(v1[i] - v2[i]) ** p

    return distance ** (1 / p)


# =========================
# Calculate Distances
# =========================

def calculateDistances(testVector, X_train, y_train, metric, p=2):

    distances = []

    for i in range(len(X_train)):

        trainVector = X_train[i]

        if metric == "euclidean":
            distance = euclideanDistance(
                testVector,
                trainVector
            )

        elif metric == "manhattan":
            distance = manhattanDistance(
                testVector,
                trainVector
            )

        elif metric == "minkowski":
            distance = minkowskiDistance(
                testVector,
                trainVector,
                p
            )

        else:
            raise ValueError(
                "Invalid metric. Choose euclidean, manhattan, or minkowski."
            )

        distances.append((distance, y_train[i]))

    return distances


# =========================
# Sorting Algorithms
# =========================

def insertionSort(data):

    for i in range(1, len(data)):

        key = data[i]
        j = i - 1

        while j >= 0 and data[j][0] > key[0]:

            data[j + 1] = data[j]
            j -= 1

        data[j + 1] = key

    return data


def bubbleSort(data):

    n = len(data)

    for i in range(n):

        swapped = False

        for j in range(0, n - i - 1):

            if data[j][0] > data[j + 1][0]:

                data[j], data[j + 1] = \
                    data[j + 1], data[j]

                swapped = True

        if not swapped:
            break

    return data


def selectionSort(data):

    n = len(data)

    for i in range(n):

        minIndex = i

        for j in range(i + 1, n):

            if data[j][0] < data[minIndex][0]:
                minIndex = j

        data[i], data[minIndex] = \
            data[minIndex], data[i]

    return data


def sortDistances(data, sortMethod):

    if sortMethod == "insertion":
        return insertionSort(data)

    elif sortMethod == "bubble":
        return bubbleSort(data)

    elif sortMethod == "selection":
        return selectionSort(data)

    else:
        raise ValueError(
            "Invalid sorting method. "
            "Choose insertion, bubble, or selection."
        )


# =========================
# KNN
# =========================

def knn(testVector, X_train, y_train, k, metric, p, sortMethod):

    # Calculate distances
    distances = calculateDistances(
        testVector,
        X_train,
        y_train,
        metric,
        p
    )

    # Sort distances
    sortedDistances = sortDistances(
        distances,
        sortMethod
    )

    # Check K
    if k < 1 or k > len(sortedDistances):

        raise ValueError(
            "K must be between 1 and "
            "the number of training samples."
        )

    # Select K nearest neighbors
    nearestNeighbors = sortedDistances[:k]

# =========================
# Distance Weighted Voting
# =========================

    votes = {}

    for distance, label in nearestNeighbors:

    # Avoid division by zero
        if distance == 0:
            weight = float("inf")
        else:
            weight = 1 / distance

        if label not in votes:
            votes[label] = weight
        else:
            votes[label] += weight


# Class with highest total weight
    predictedLabel = max(votes, key=votes.get)

    return predictedLabel    
# =========================
# KNN Main Program
# =========================

metric = "euclidean"
p = 2
sortMethod = "insertion"

# Store performance for every K
results = []

# K from 1 to number of test samples
for k in range(1, len(X_test) + 1):

    predictions = []

    # Predict every test sample
    for testVector in X_test:

        predictedLabel = knn(
            testVector,
            X_train,
            y_train,
            k,
            metric,
            p,
            sortMethod
        )

        predictions.append(predictedLabel)

    # =========================
    # Calculate TP, TN, FP, FN
    # =========================

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    # Assuming:
    # 1 = Schizophrenia
    # 0 = Healthy
    #
    # If your labels are strings, change these accordingly.

    for i in range(len(y_test)):

        actual = y_test[i]
        predicted = predictions[i]

        if actual == 1 and predicted == 1:
            TP += 1

        elif actual == 0 and predicted == 0:
            TN += 1

        elif actual == 0 and predicted == 1:
            FP += 1

        elif actual == 1 and predicted == 0:
            FN += 1

    # =========================
    # Calculate Metrics
    # =========================

    total = TP + TN + FP + FN

    accuracy = (TP + TN) / total

    if TP + FP != 0:
        precision = TP / (TP + FP)
    else:
        precision = 0

    if TP + FN != 0:
        recall = TP / (TP + FN)
    else:
        recall = 0

    if precision + recall != 0:
        f1Score = (
            2 * precision * recall
        ) / (precision + recall)
    else:
        f1Score = 0

    # Store results
    results.append(
        (
            k,
            accuracy,
            precision,
            recall,
            f1Score
        )
    )


# =========================
# Display Results
# =========================

print("\nKNN Performance for Different K Values")
print("-" * 70)

print(
    f"{'K':<5}"
    f"{'Accuracy':<15}"
    f"{'Precision':<15}"
    f"{'Recall':<15}"
    f"{'F1 Score':<15}"
)

print("-" * 70)

for k, accuracy, precision, recall, f1Score in results:

    print(
        f"{k:<5}"
        f"{accuracy:<15.4f}"
        f"{precision:<15.4f}"
        f"{recall:<15.4f}"
        f"{f1Score:<15.4f}"
    )