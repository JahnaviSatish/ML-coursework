import pandas as pd
df=pd.read_excel('ML-lab2 dataset.xlsx',sheet_name=('marketing_campaign'))
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"]) 
#date is not categorical but it is stored as a string
#from A1 after identifying that date is stored as string we change it to this

def labelEncoding(df, column):
    data = df.copy()# Making a copy so the original dataframe doesn't change
    values = data[column].dropna().unique()#all unique values in the column
    labels = {}    # Dictionary- storing category and number

    for i, value in enumerate(values): # category starting from 0
        labels[value] = i
    # Replace text values with numbers
    data[column] = data[column].map(labels)
    return data, labels

def oneHot(df, column):
    data = df.copy()
    values = data[column].dropna().unique() # same as above
    # Create a new column for each category
    for value in values:
        new_column = column + "_" + str(value)
        data[new_column] = (data[column] == value).astype(int)# 1 if the row is in category else 0
    data.drop(column, axis=1, inplace=True)   # Remove the old text column
    return data

df1 = df.copy()#making df copy so that encoding can be stored here
ordinal = ["Education"]   # Label Encoding identified from A1
nominal = ["Marital_Status"]   # One-Hot Encoding identified from a1

for col in ordinal:
    df1, mapping = labelEncoding(df1, col)
    print(col, mapping)

for col in nominal:
    df1 = oneHot(df1, col)

print(df1.head())
#to check dimentionalty before and after we use shape
#indicates the number of elements along each dimension or axis
print("original dimension=",df.shape)
print("encoded df dimension:", df1.shape)
        