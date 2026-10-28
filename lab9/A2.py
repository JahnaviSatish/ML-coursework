from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
import pandas as pd

df = pd.read_csv("eeg_features.csv")
f_cols = [c for c in df.columns if c not in ["subject", "label"]]
X = df[f_cols].values.tolist()
y = df["label"].tolist()
param_dist = {
    "hidden_layer_sizes": [(5,), (10,), (20,), (10, 5)],
    "activation": ["relu", "tanh", "logistic"],
    "learning_rate_init": [0.001, 0.01, 0.05, 0.1],
    "max_iter": [500, 1000]
}
random_search = RandomizedSearchCV(
    estimator=MLPClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=10,       # how many random combinations to try
    cv=5,
    scoring="accuracy",
    random_state=42
)

random_search.fit(X, y)
print("Best parameters:", random_search.best_params_)
print("Best cross-validated acc:", random_search.best_score_)