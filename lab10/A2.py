import pandas as pd, seaborn as sns, matplotlib.pyplot as plt

SAVE = r"C:\Users\User\OneDrive\clg work\ML-coursework\lab10\feature_correlation_heatmap.png"

df = pd.read_csv("features_final_16.csv")
X = df.drop(columns=["subject", "label"])

plt.figure(figsize=(11, 9))
sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation-16 microstate features")
plt.tight_layout()
plt.savefig(SAVE, dpi=160)
plt.show()