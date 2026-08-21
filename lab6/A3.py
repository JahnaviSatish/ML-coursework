import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
df=pd.read_csv("eeg_features.csv")  
X = df.drop(columns=["label","subject"]).copy()
y = df["label"].copy()
print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)  
#random_state=42  same split every time you run program
# stratify=y --> healthy/schizophrenia class proportions are maintained in both sets.

#### A4 ######

results = []

for k in range(1, 10):

    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)

    ####### A5 #######
    accuracy = neigh.score(X_test, y_test)

    ####### A6 ########
    prediction = neigh.predict(X_test)

    # =========================
    # Calculate TP, TN, FP, FN
    # =========================

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(len(y_test)):

        actual = y_test.iloc[i]
        predicted = prediction[i]

        if actual == 1 and predicted == 1:
            TP += 1

        elif actual == 0 and predicted == 0:
            TN += 1

        elif actual == 0 and predicted == 1:
            FP += 1

        elif actual == 1 and predicted == 0:
            FN += 1

    # =========================
    # Calculate Metrics
    # =========================

    total = TP + TN + FP + FN

    accuracy = (TP + TN) / total

    if TP + FP != 0:
        precision = TP / (TP + FP)
    else:
        precision = 0

    if TP + FN != 0:
        recall = TP / (TP + FN)
    else:
        recall = 0

    if precision + recall != 0:
        f1Score = (
            2 * precision * recall
        ) / (precision + recall)
    else:
        f1Score = 0

    # Store results
    results.append(
        (
            k,
            accuracy,
            precision,
            recall,
            f1Score
        )
    )

# =========================
# Display Results
# =========================

print("\nKNN Performance for Different K Values")
print("-" * 70)

print(
    f"{'K':<5}"
    f"{'Accuracy':<15}"
    f"{'Precision':<15}"
    f"{'Recall':<15}"
    f"{'F1 Score':<15}"
)

print("-" * 70)

for k, accuracy, precision, recall, f1Score in results:

    print(
        f"{k:<5}"
        f"{accuracy:<15.4f}"
        f"{precision:<15.4f}"
        f"{recall:<15.4f}"
        f"{f1Score:<15.4f}"
    )