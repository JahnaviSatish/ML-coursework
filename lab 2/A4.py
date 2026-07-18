import pandas as pd
import numpy as np

df = pd.read_excel("ML-lab2 dataset.xlsx", sheet_name="thyroid0387_UCI",na_values=['?'])# this is to treat ? as NaN values for correct output
def identify():
    categorical = df.select_dtypes(include='object').columns
    numerical = df.select_dtypes(include=['int64','float64']).columns
    return categorical,numerical

def missingvalues():
    return df.isnull().sum()

def outlier():
    result = []
    for col in numerical:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower) | (df[col] > upper)]
        result.append(len(outliers))

    return result

categorical,numerical=identify()
count=missingvalues()
df.info()#searching each attribute for dtype

print("Categorical:")
print(categorical)
print("\nNumerical:")
print(numerical)

for col in categorical:
    print("\nAttribute :", col)
    print("Unique values :", df[col].dropna().unique())

    if len(df[col].dropna().unique()) == 2:
        print("Encoding : Label Encoding")
    else:
        print("Encoding : One-Hot Encoding")

print(df[numerical].describe())# study data range


print(count)

c = outlier()

for col, x in zip(numerical, c):
    print(col, ":", x, "outliers")

for col in numerical:
    print(col)
    print("Mean :",df[col].mean())
    print("Variance :",df[col].var())
    print("Standard Deviation :",df[col].std())
    print()