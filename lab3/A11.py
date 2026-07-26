import numpy as np
import random
import pandas as pd
df=pd.read_excel('ML-lab2 dataset.xlsx',sheet_name=('marketing_campaign'))
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"]).astype("int64")
df1 = df.copy()

def labelEncoding(df, column):
    data = df.copy()
    values = data[column].dropna().unique()
    labels = {}  
    for i, value in enumerate(values): 
        labels[value] = i
    data[column] = data[column].map(labels)
    return data, labels

def oneHot(df, column):
    data = df.copy()
    values = data[column].dropna().unique()
    for value in values:
        new_column = column + "_" + str(value)
        data[new_column] = (data[column] == value).astype(int)
    data.drop(column, axis=1, inplace=True)  

    return data

# used to find the nearest centroid
def euclidean(v1,v2):
    euclidean=0
    for i,j in zip(v1,v2):
        euclidean+=(j-i)**2
    euclidean=np.sqrt(euclidean)
    return euclidean


# Func to calculate centroid of one cluster
# The centroid is = the average of all points in that cluster
def centroid(points):
    if len(points) == 0:
        return None#cluster becomes empty, return None
    centre = []
    # Finding average of every column
    # Every column represents one feature
    for i in range(len(points[0])):
        total = 0
        # Add all values of one feature
        for point in points:
            total += point[i]
        centre.append(total / len(points))

    return np.array(centre)

def kmeans(data, k):
    centroids = []
    index = random.sample(range(len(data)), k)

    for i in index:
        centroids.append(data[i])

    while True:
        clusters = []
        for point in data:    # Assign every point to the nearest centroidd
            dist = []
            # Find distance from current point
            # to every centroid
            for centre in centroids:
                dist.append(euclidean(point, centre))
            # Find the nearest centroid
            nearest = dist.index(min(dist))
            clusters.append(nearest)# Storing cluster number
        # Recalc the centroid of every cluster
        new_centroids = []
        for i in range(k):#one cluster at a time
            points = []

            # Collect all points belonging to the current cluster
            for point, c in zip(data, clusters):
                if c == i:
                    points.append(point)
            # If no points belong to the cluster,
            # keep the previous centroid
            if len(points) == 0:
                new_centroids.append(centroids[i])
            else:# Otherwise calculate the new centroid
                new_centroids.append(centroid(points))

        stop = True# Stop when centroids dont change
        for old, new in zip(centroids, new_centroids):
            # Compare old centroid with newly calculated centroid
            if not np.allclose(old, new):
                stop = False
                break
        if stop:# Exit the loop if the centroids remain the same
            break

        # Otherwise continue with the new centroids
        centroids = new_centroids
    return clusters, centroids

df1 = df1.fillna(df1.median(numeric_only=True))
ordinal = ["Education"]   # Education has an order, so Label Encoding is used
nominal = ["Marital_Status"] # Marital Status has no order, so One-Hot Encoding is used

for col in ordinal:
    df1, mapping = labelEncoding(df1, col)
for col in nominal:
    df1 = oneHot(df1, col)
# Converting dataframe into a matrix
# K-Means works with numerical matrices

data = df1.to_numpy(dtype=float)
k = 3# Number of clusters to be formed
clusters, centroids = kmeans(data, k)
print("Cluster for every data point")
print(clusters)
print()
print("Final Centroids")
for c in centroids:
    print(c)