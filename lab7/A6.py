from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("eeg_features.csv")
feature_cols = [c for c in df.columns if c not in ["subject", "label"]]
X = df[feature_cols]
y = df["label"].tolist()
model = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
model.fit(X, y)
plt.figure(figsize=(16, 10))
plot_tree(
    model,
    feature_names=feature_cols,
    class_names=["healthy", "schizophrenia"],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("DT(entropy criterion, max_depth=3)")
plt.show()