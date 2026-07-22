import pandas as pd
df=pd.read_excel('ML-lab2 dataset.xlsx',sheet_name=('marketing_campaign'))
#print(df)
#studying feature vectors
## after studying feature vectors, categorize them into nominal,ordinal,ratio,interval here later
def identify():
    categorical=df.select_dtypes(include="object").columns
    numerical=df.select_dtypes(include=['int64','float64']).columns
    return categorical,numerical
df.info()# same output as identofy() but as list and not included in excel
categoric,numeric=identify()
print("Categorical:")
print(categoric)
print("\nNumerical:")
print(numeric)
