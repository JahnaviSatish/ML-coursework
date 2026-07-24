from scipy.spatial import distance
import pandas as pd
df=pd.read_excel('ML-lab2 dataset.xlsx',sheet_name=('marketing_campaign'))
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"]).astype("int64")# making date as an integer or have to drop column
df1 = df.copy()

def minkowski_dist(v1,v2,p):
    dist=0
    for i,j in zip(v1,v2):
        dist+=(abs(i-j))**p
    dist=dist**(1/p)
    return dist

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


ordinal = ["Education"]   
nominal = ["Marital_Status"]  

for col in ordinal:
    df1, mapping = labelEncoding(df1, col)
    print(col, mapping)

for col in nominal:
    df1 = oneHot(df1, col)

print(df1.head())
A=df1.iloc[0].to_numpy()
B=df1.iloc[1].to_numpy()
#till here same code as A2&A3..refer there for understanding code
print(A)
print(B) 

for p in range(1, 11):
    own = minkowski_dist(A, B, p)
    scipy = distance.minkowski(A, B, p)

    print("p =", p)
    print("Own:", own)
    print("SciPy:", scipy)
