#normalisation happens to only numeric data
# and from previous results its about 8 cols 
# 7 if we exclude record id as it is an identifier
import pandas as pd
import numpy as np

df = pd.read_excel("ML-lab2 dataset.xlsx", sheet_name="thyroid0387_UCI", na_values=['?'])
def identify():
    categorical = df.select_dtypes(include='object').columns
    numerical = df.select_dtypes(include=['int64','float64']).columns
    if 'Record ID' in numerical:
        numerical = numerical.drop('Record ID') 
    return categorical,numerical

def normalize(numerical):
    normalized = df.copy() #again, creating a copy to avoid changing original df
    for col in numerical:
        mini= df[col].min() #each numeric column, finding min and max value
        maxi = df[col].max()
        normalized[col] = (df[col] - mini) / (maxi - mini)
        # normalized col for normalized values, formula= (x-min(x)) / max(x)-min(x)​
    return normalized

categorical,numerical=identify()
print("\nNumerical:")
print(numerical)
normalized = normalize(numerical)
print(normalized)