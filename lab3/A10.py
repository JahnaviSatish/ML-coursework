import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_excel('ML-lab2 dataset.xlsx', sheet_name='marketing_campaign')
df1=df.copy()
feature = df1["Recency"].dropna()#choosing recency for histogram and removing missing vals
hist,bins = np.histogram(feature, bins=15)
#taken from A8 and A9
def mean(x):
    total = 0
    count = 0
    for i in x:
        total+=i
        count+=1
    return total/count

def variance(x):
    m = mean(x)
    total = 0
    count = 0
    for i in x:
        total+=(i-m)**2
        count+=1
    return total/count

M=mean(feature)
V=variance(feature)
print("mean=",M)
print("variance=",V)
print("histogram values:",hist)
print("bin ranges:",bins)
plt.hist(feature, bins=15, edgecolor="black")
plt.xlabel("Recency")
plt.ylabel("Frequency")
plt.title("Histogram of Recency")
plt.show()
