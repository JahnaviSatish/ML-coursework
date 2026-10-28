from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd

df = pd.read_csv("eeg_features.csv")
feature_cols = [c for c in df.columns if c not in ["subject", "label"]]
X = df[feature_cols].values.tolist()
y = df["label"].tolist()

classifiers = {
    "SVM": SVC(),
    "Decision Tree":DecisionTreeClassifier(random_state=42),
    "Random Forest":RandomForestClassifier(random_state=42),
    "AdaBoost":AdaBoostClassifier(random_state=42),
    "Naive Bayes":GaussianNB()
}
# names in the classifiers , inside brackets should be the exact same name. otherwise error

results = []
for name, clf in classifiers.items():
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    results.append({"Classifier": name, "Mean Accuracy": scores.mean(), "Std Dev": scores.std()})
results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv("comparison.csv", index=False)
