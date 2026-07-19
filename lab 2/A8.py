# same logic as my A4 code for detecting outliers

import pandas as pd
import numpy as np

df = pd.read_excel("ML-lab2 dataset.xlsx", sheet_name="thyroid0387_UCI", na_values=['?'])

def identify():
    categorical = df.select_dtypes(include='object').columns
    numerical = df.select_dtypes(include=['int64', 'float64']).columns
    return categorical, numerical

def outlier():
    outlierColumns = []

    for col in numerical:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        if len(outliers) > 0:
            outlierColumns.append(col)
    return outlierColumns

def imputation(): #main logic which decides whether na values are filled with mean,median or mode
    outlierColumns = outlier()
    for col in numerical:
        if col in outlierColumns:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mean())
    for col in categorical:
        df[col] = df[col].fillna(df[col].mode()[0])
# also not displaying what values are being filled in each missing value cell 
# as there are too many values to show
# but it can be shown

categorical,numerical = identify()
imputation()
print("missing values after imputation=")
print(df.isnull().sum())# this shows how many missing values are left ,which should be 0 after imputation
outlierColumns = outlier()

for col in numerical:
    if col in outlierColumns:
        print(col, ": Median")
    else:
        print(col, ": Mean")
for col in categorical:
    print(col, ": Mode")