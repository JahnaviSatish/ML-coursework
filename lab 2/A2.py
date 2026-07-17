import pandas as pd
from sklearn.tree import DecisionTreeClassifier
df=pd.read_excel("ML-lab2 dataset.xlsx",sheet_name="Purchase data")
def feature():
    x=df[['Candies (#)','Mangoes (Kg)','Milk Packets (#)']].to_numpy()
    return x
def classify():
    df["class"]=df["Payment (Rs)"].apply(lambda x:"rich" if x>200 else "poor")
    return df["class"]
def train(X,Z):
    model=DecisionTreeClassifier(random_state=11)#random_state=random number generator so that the tree stays the same every time code is run
    model.fit(X,Z)#Without random_state, running the same code many times may give slightly different trees.
    return model 
def predict(res,X):
    df["Prediction"] = res.predict(X)   
    return df["Prediction"]

X=feature()
Z=classify()
res=train(X,Z)# training model with feature data and classify instead of payment so it can learn and make predictions later
pred=predict(res,X)
print(df[['Customer','Candies (#)','Mangoes (Kg)','Milk Packets (#)','Payment (Rs)','class','Prediction']])