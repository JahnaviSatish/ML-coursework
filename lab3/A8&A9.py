import numpy as np
import pandas as pd
df = pd.read_excel('ML-lab2 dataset.xlsx', sheet_name='marketing_campaign')
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"]).astype("int64")#date into numeric form
df1 = df.copy()
#taken from before
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

def sd(x):
    return np.sqrt(variance(x))
def statistics(df):
    result = {}
    for column in df.columns:
        # Removing missing values
        A = df[column].dropna().to_numpy()
        result[column] = {
            "Mean": mean(A),
            "Variance": variance(A),
            "Std Dev": sd(A)
        }
    return result

ordinal = ["Education"]
nominal = ["Marital_Status"]
for col in ordinal:# taken from before
    df1, mapping=labelEncoding(df1, col)
for col in nominal:
    df1=oneHot(df1, col)
s = statistics(df1)
for col in s:
    print(col)
    print("Mean=",s[col]["Mean"])
    print("Variance=",s[col]["Variance"])
    print("Standard Deviation=",s[col]["Std Dev"])
    print()

#A9 from here
# Mean and Standard Deviation using NumPy
mean_numpy = np.mean(df1, axis=0)
std_numpy = np.std(df1, axis=0)
#axis = 0 ↓ → move down the columns (calculate one value per column)
#axis = 1 → → move across the rows (calculate one value per row)
# axis understanding taken from internet, here axis is 0 because feature vector was asked
print("Comparison A9")
for col in df1.columns:
    print(col)
    print("Own Mean =", s[col]["Mean"])#same as before,calling it again
    print("NumPy Mean =", mean_numpy[col])
    print()
    print("Own Standard Deviation =", s[col]["Std Dev"])
    print("NumPy Standard Deviation =", std_numpy[col])
    print()