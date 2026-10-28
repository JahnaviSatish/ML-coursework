import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
df=pd.read_excel('ML-lab2 dataset.xlsx',sheet_name=('marketing_campaign'))
df=df.drop(columns='Dt_Customer')
print(df.shape)
le = LabelEncoder()
oh= OneHotEncoder(sparse_output=False)
df['Education']=le.fit_transform(df['Education'])
encoded=oh.fit_transform(df[['Marital_Status']])
df1=pd.DataFrame(encoded,columns=oh.get_feature_names_out(['Marital_Status']),index=df.index)
df = df.drop(columns=['Marital_Status'])
df = pd.concat([df, df1], axis=1)
print(df.dtypes)
print(df.shape)

Q1 = df['Income'].quantile(0.25)
Q3 = df['Income'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

upper_array = np.where(df['Income'] >= upper)[0]
lower_array = np.where(df['Income'] <= lower)[0]

df.drop(index=upper_array, inplace=True)
df.drop(index=lower_array, inplace=True)

df['Income'] = df['Income'].fillna(np.mean(df['Income']))
print(df.isnull().sum())

scaler = MinMaxScaler()
scaled_df=scaler.fit_transform(df)
inertias = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_df)
    inertias.append(kmeans.inertia_)

print(inertias)

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(scaled_df)

print("Cluster labels:")
print(kmeans.labels_)

print("\nCluster centroids:")
print(kmeans.cluster_centers_)