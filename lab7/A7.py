import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = pd.read_csv("eeg_features.csv")

feat1, feat2 = "state2_duration", "trans_1to2"
X_2d = df[[feat1, feat2]].values
y_arr = df["label"].values

clf_2d = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
clf_2d.fit(X_2d, y_arr)

x_min, x_max = X_2d[:, 0].min()-0.01,X_2d[:, 0].max() + 0.01
y_min, y_max = X_2d[:, 1].min()-0.01,X_2d[:, 1].max() + 0.01
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_arr, cmap="coolwarm", edgecolors='k')
plt.xlabel(feat1)
plt.ylabel(feat2)
plt.title(f"Decision Boundary: {feat1} vs {feat2}")
plt.show()