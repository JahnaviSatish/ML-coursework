import numpy as np
import random
import pandas as pd
df=pd.read_excel('ML-lab2 dataset.xlsx',sheet_name=('marketing_campaign'))
# for now i am dropping all non numeric columns
df1 = df.select_dtypes(include=['number']).dropna().to_numpy()

# used to find the nearest centroid
def euclidean(v1,v2):
    euclidean=0
    for i in range(len(v1)):
        euclidean+=(v1[i]-v2[i])**2
    euclidean=np.sqrt(euclidean)
    return euclidean

# The centroid is = the average of all points in that cluster
# randomly initializing k centroid by picking k samples from df1
def initialize_random_centroids(k, df1):
    m = len(df1)
    n = len(df1[0])
    # a centroid should be of shape (1,n), so the centroids array will be of shape (k,n)
    centroids = []
    for i in range(k):
        a = random.randint(0, m - 1)
        centroids.append(df1[a])
    return centroids

def closest_centroid(df1, centroids, k):
    distance=[]
    closest=0
    mindist= euclidean(df1, centroids[0]) #distance to the first centroid
    for i in range(k):
        distance.append(euclidean(centroids[i], df1))
         # If this centroid is closer, update the minimum distance
        if distance[i] < mindist:
            mindist = distance[i]
            closest = i

    # Return the indedf1 of the closest centroid
    return closest

def create_clusters(centroids, k, df1):
    m = len(df1)
    cluster_idx = [] #empty list to store clusters
    for i in range(m):
        #Find the indedf1 of the closest centroid for the current data point
        closest = closest_centroid(df1[i], centroids, k)
        #cluster indedf1 in list
        cluster_idx.append(closest)
    return cluster_idx

# new centroids calc
def new_centroids(cluster_idx, k, df1):
    centroids = []
    for i in range(k): #say 0,1,2
        points = []
        for j in range(len(df1)): # say 2200 patients
            if cluster_idx[j] == i: # see how many cluster indedf1 for all 2200 patients match k values 0,1,2
                points.append(df1[j]) # if they match put that in points array as a feature vector
        centroids.append(np.mean(points, axis=0)) # finding the average of all the vectors collected in 1 cluster , axis 0 means column wise
        # example point=[[20,10],[15,5]] then centroids = (20+15 / 2 , 10+5 / 2)
    return np.array(centroids) # send an array of new centroids calculated for k=0,1,2
    
def kmeans(k, df1, iterations):
    centroids = initialize_random_centroids(k, df1) #step1 random k centers
    for i in range(iterations):
        print("Iteration:", i + 1)
        clusters = create_clusters(centroids, k, df1) #step2 forming clusters
        old_centroids = centroids
        centroids = new_centroids(clusters, k, df1) #step3 new centroids
        if np.array_equal(old_centroids, centroids): #step4 convergence checking
            break
    return centroids

import time
k = 3# Number of clusters to be formed
start= time.time()
c = kmeans(k,df1,100)
end=time.time()
t=end-start
print("Centroids:")
print(c)
print(t)

