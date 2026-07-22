import pandas as pd
df=pd.read_excel('ML-lab2 dataset.xlsx',sheet_name=('marketing_campaign'))
def identify():
    categorical=df.select_dtypes(include="object").columns
    numerical=df.select_dtypes(include=['int64','float64']).columns
    return categorical,numerical

def encoding(categoric):
    for col in categoric:
        count=df[col].dropna().unique()#df[col].dropna().unique()= says in each column, after dropping na values,
        # taking only unique values. same note from lab2
        if len(df[col].dropna().unique()) == 2:
            result="label encoding"
        else:
            result="one-hot encoding"
    return result

categoric,numeric=identify()
result=encoding(categoric)
print("Categorical:")
print(categoric)
for col in categoric:
    print("\nAttribute :", col)
    print("encoding=",result)
    
    
        