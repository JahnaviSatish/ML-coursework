import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_excel('ML-lab2 dataset.xlsx', sheet_name='marketing_campaign')
df = df.dropna()
# modular code by chatgpt
# Select Recency Feature
recency = df["Recency"]
# Sample Income Values test case
income = [12000, 15000, 18000, 18000, 22000,
          25000, 27000, 30000, 30000, 35000]

# Generate Histogram Data
frequency1, bins1 = np.histogram(income, bins=5)
print("Bins(test case):", bins1)
print("Frequency(test case):", frequency1)
# Generate Histogram Data
frequency, bins = np.histogram(recency, bins=10)
print("Bins(actual data):", bins)
print("Frequency(actual data):", frequency)

# Plot Histogram
plt.hist(income, bins=5, edgecolor='black')
plt.title("Histogram of Income(test case)")
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.grid(True)

plt.show()

# Plot Histogram
plt.hist(recency, bins=10, edgecolor='black')

plt.title("Histogram of Recency")
plt.xlabel("Recency")
plt.ylabel("Frequency")
plt.grid(True)

plt.show()