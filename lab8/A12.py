import pandas as pd
from sklearn.neural_network import MLPClassifier

df = pd.read_csv("lab8/eeg_features.csv")
X = df.drop(columns=["label","subject"])
y = df["label"]

clf = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5, 2),random_state=1)

clf.fit(X, y)
predictions = clf.predict(X)
print("Predictions:", predictions)
print("Actual:",y)