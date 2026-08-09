import numpy as np
import random
import math
import pandas as pd
df=pd.read_excel('ML-lab2 dataset.xlsx',sheet_name=('marketing_campaign'))
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"]).astype("int64")
df1 = df.select_dtypes(include=['number']).dropna().to_numpy()

def euclidean(v1, v2):

    assert len(v1) == len(v2)

    distance = 0

    for i in range(len(v1)):
        distance += (v1[i] - v2[i]) ** 2

    return math.sqrt(distance)

def initializeCentroids(data, k):
    # Randomly select k unique indices from the dataset
    indices = random.sample(range(len(data)), k)

    # Select the corresponding data points as the initial centroids
    centroids = data[indices]

    # Return the selected centroids
    return np.array(centroids)


def calculateCentroids(data, labels, k):
    # List to store the new centroids
    centroids = []

    # Calculate one centroid for each cluster
    for cluster in range(k):

        # Select all data points that belong to the current cluster
        cluster_points = data[labels == cluster]

        # If the cluster contains points, compute the mean of each feature
        if len(cluster_points) > 0:
            centroid = np.mean(cluster_points, axis=0)

        # If no points are assigned to the cluster, keep a zero vector
        # (This avoids errors due to empty clusters.)
        else:
            centroid = np.zeros(data.shape[1])

        # Store the computed centroid
        centroids.append(centroid)

    # Convert the list of centroids into a NumPy array
    return np.array(centroids)

def assignClusters(data, centroids):
    # List to store the cluster assigned to each data point
    labels = []

    # Iterate through each data point
    for point in data:

        # List to store distances from the current point to each centroid
        distances = []

        # Compute the Euclidean distance to every centroid
        for centroid in centroids:
            distance = euclidean(point, centroid)
            distances.append(distance)

        # Assign the point to the nearest centroid
        labels.append(np.argmin(distances))

    # Return cluster labels as a NumPy array
    return np.array(labels)

def converged(oldCentroids, newCentroids, tolerance):
    # Compute the Euclidean distance moved by each centroid
    distances = np.sqrt(np.sum((oldCentroids - newCentroids) ** 2, axis=1))

    # Check if every centroid has moved less than the tolerance
    return np.all(distances < tolerance)

def kMeans(data, k, max_iterations):

    # Randomly initialize the centroids
    centroids = initializeCentroids(data, k)

    # Repeat until convergence or maximum iterations
    for _ in range(max_iterations):

        # Assign each data point to the nearest centroid
        labels = assignClusters(data, centroids)

        # Compute the new centroids
        newCentroids = calculateCentroids(data, labels, k)

        # Check for convergence
        if converged(centroids, newCentroids, tolerance=1e-4):
            centroids = newCentroids
            break

        # Update centroids for the next iteration
        centroids = newCentroids

    # Return the final cluster assignments and centroids
    return labels, centroids
######################################################################################

# Number of clusters
k = 3

# Maximum iterations
max_iterations = 100

# Run K-Means
labels, centroids = kMeans(df1, k, max_iterations)

print("Cluster Labels:")
print(labels)

print("\nFinal Centroids:")
print(centroids)

########## performance ##########
import time
import tracemalloc

tracemalloc.start()

start = time.time()

labels, centroids = kMeans(df1, k=3, max_iterations=100)

end = time.time()

current, peak = tracemalloc.get_traced_memory()

tracemalloc.stop()

print("Execution Time:", end - start, "seconds")
print("Peak Memory Usage:", peak / 1024, "KB")