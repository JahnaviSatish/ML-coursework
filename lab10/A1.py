import os
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score

 

OUTPUT_DIR = r"C:\Users\User\OneDrive\clg work\ML-coursework\lab10"
TABLES = {"with s14": "features_final_16.csv", "without s14": "features_final_16_no_s14.csv"}
SEED = 42
INNER_FOLDS = 3          
OUT_FILE = "lab10_a1_stacking_results.csv"

warnings.filterwarnings("ignore")
 
 
def scaled(model):
    return make_pipeline(StandardScaler(), model)
 
 
def base_models():
    """The base classifiers (name, model). Replace by the classifier panel when it is ready."""
    return [
        ("logreg", scaled(LogisticRegression(max_iter=1000, random_state=SEED))),
        ("svm", scaled(SVC(kernel="rbf", probability=True, random_state=SEED))),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=SEED)),
        ("knn", scaled(KNeighborsClassifier(n_neighbors=5))),
        ("nb", GaussianNB()),
    ]
 
 
def meta_models():
    """The final_estimators to compare."""
    return {
        "meta: logistic regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "meta: random forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
        "meta: SVM": SVC(probability=True, random_state=SEED),
        "meta: kNN": KNeighborsClassifier(n_neighbors=5),
    }
 
 
def make_stack(final_estimator):
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
    return StackingClassifier(estimators=base_models(), final_estimator=final_estimator,
                              cv=inner, stack_method="predict_proba")
 
 
def wilson_ci(k, n, z=1.96):
    p = k / n
    centre = (p +z *z /(2 * n))/(1+z*z/n)
    half = z * np.sqrt(p * (1 - p) / n+z*z / (4*n*n)) / (1 + z*z /n)
    return centre - half, centre+half
 
 
def loso(model, X, y):
    """Leave-one-subject-out: one row = one subject. Returns the predictions."""
    pred = np.zeros(len(y), dtype=int)
    for train, test in LeaveOneOut().split(X):
        m = clone(model).fit(X[train], y[train])
        pred[test] = m.predict(X[test])
    return pred
 
 
rows = []
for table_name, fname in TABLES.items():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, fname))
    features = [c for c in df.columns if c not in ("subject", "label")]
    X, y = df[features].to_numpy(float), df["label"].to_numpy(int)
    models = {f"single: {n}": m for n, m in base_models()}
    models.update({name: make_stack(fe) for name, fe in meta_models().items()})
    print(f"\n=== {table_name}: {len(df)} subjects, {len(features)} features ===")
    for name, model in models.items():
        pred = loso(model, X, y)
        acc = accuracy_score(y, pred)
        bacc = balanced_accuracy_score(y, pred)
        lo, hi = wilson_ci(int((pred == y).sum()), len(y))
        rows.append({"table": table_name, "model": name, "accuracy": acc,
                     "balanced_accuracy": bacc, "ci_low": lo, "ci_high": hi})
        print(f"  {name:28s} acc {acc:.3f}  bal.acc {bacc:.3f}  95% CI [{lo:.2f}, {hi:.2f}]")
 
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUTPUT_DIR, OUT_FILE), index=False)
print(f"\nSaved outputs/{OUT_FILE}")
print("chance level is 0.50. with 28 subjects, differences of few points are within noise.")