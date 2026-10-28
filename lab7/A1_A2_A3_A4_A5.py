import pandas as pd
import numpy as np


df = pd.read_csv("eeg_features.csv")
print(df['label'].value_counts())

def calc_entropy(labels):
    counts = pd.Series(labels).value_counts()
    probabilities = counts / len(labels)
    
    entropy = 0
    for p in probabilities:
        entropy += -p * np.log2(p)
    
    return entropy

def calc_gini(labels):
    counts = pd.Series(labels).value_counts()
    probabilities = counts/len(labels)
    
    gini = 1
    for p in probabilities:
        gini -= p ** 2
    
    return gini

def bin_feature(values, bin_type="equal_width", n_bins=4):
    if bin_type == "equal_width":
        min_val = min(values)
        max_val = max(values)
        data_range = max_val - min_val

        if data_range == 0:
            return [0] * len(values)
        width = data_range / n_bins
        bin_assignments = []

        for x in values:
            shifted_value = x - min_val
            bin_idx = int(shifted_value / width)
            if bin_idx >= n_bins:
                bin_idx = n_bins - 1
            bin_assignments.append(bin_idx)

        return bin_assignments

    elif bin_type == "equal_freq":
        bin_assignments = pd.qcut(values, q=n_bins, labels=False, duplicates="drop")
        return list(bin_assignments)

    
def select_root_feature(X, y, bin_type="equal_width", n_bins=4):
    feature_names = X.columns.tolist()

    
    original_entropy = calc_entropy(y)

    best_feature = None
    best_gain = -1  

    for feature in feature_names:

        values = X[feature].tolist()
        bins = bin_feature(values, bin_type=bin_type, n_bins=n_bins)

        unique_bins = list(set(bins))

        weighted_entropy = 0

        for b in unique_bins:
            
            labels_in_bin = []
            for i in range(len(bins)):
                if bins[i] == b:
                    labels_in_bin.append(y[i])

           
            bin_entropy = calc_entropy(labels_in_bin)

            
            weight = len(labels_in_bin) / len(y)

            weighted_entropy += weight * bin_entropy

        
        info_gain = original_entropy - weighted_entropy
        
        if info_gain > best_gain:
            best_gain = info_gain
            best_feature = feature

    return best_feature, best_gain

feature_cols = [c for c in df.columns if c not in ["subject", "label"]]
X = df[feature_cols]
y = df["label"].tolist()

print("Entropy of dataset:", calc_entropy(y))
print("Gini of dataset:", calc_gini(y))

best_feature, best_gain = select_root_feature(X, y)
print("Best root feature:", best_feature)
print("Information gain:", best_gain)


def majority_class(labels):
    counts = pd.Series(labels).value_counts()
    return counts.idxmax()   

def build_tree(X, y, bin_type="equal_width", n_bins=4, depth=0, max_depth=3):

    
    if len(set(y)) == 1:
        return {"leaf": True, "prediction": y[0]}

   
    if depth >= max_depth:
        return {"leaf": True, "prediction": majority_class(y)}

    
    if len(y) < 2:
        return {"leaf": True, "prediction": majority_class(y)}

   
    best_feature, best_gain = select_root_feature(X, y, bin_type=bin_type, n_bins=n_bins)

    
    if best_gain <= 0:
        return {"leaf": True, "prediction": majority_class(y)}

   
    values = X[best_feature].tolist()
    bins = bin_feature(values, bin_type=bin_type, n_bins=n_bins)

    branches = {}
    unique_bins = list(set(bins))

    for b in unique_bins:
        X_subset_rows = []
        y_subset = []

        for i in range(len(bins)):
            if bins[i] == b:
                X_subset_rows.append(X.iloc[i])
                y_subset.append(y[i])

        X_subset = pd.DataFrame(X_subset_rows)

        
        branches[b] = build_tree(
            X_subset, y_subset,
            bin_type=bin_type, n_bins=n_bins,
            depth=depth + 1, max_depth=max_depth
        )

    return {"feature": best_feature, "branches": branches}

tree = build_tree(X, y, bin_type="equal_width", n_bins=4, max_depth=3)
print(tree)

# for my reference
# Level 1 — the root
# {'feature': 'trans_1to2', 'branches': {0: ..., 1: ..., 2: ..., 3: ...}}

# inside Branch 0 
# {'feature': 'trans_1to2', 'branches': {0: ..., 1: {'prediction': 1}, 2: {'prediction': 0}, 3: {'prediction': 0}}}

# the very last leaves
# 3: {'leaf': True, 'prediction': np.int64(0)}