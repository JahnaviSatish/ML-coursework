from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

df = pd.read_csv("eeg_features.csv")

feature_cols = [c for c in df.columns if c not in ["subject", "label"]]
X = df[feature_cols]
y = df["label"].tolist()

param_grid = {
    "max_depth": [1, 2, 3, 4, 5],
    "criterion": ["entropy", "gini"],
    "min_samples_split": [2, 3, 4]
}

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy"
)
grid_search.fit(X, y)
print("Best parameters:", grid_search.best_params_)
print("Best cross-validated acc:", grid_search.best_score_)