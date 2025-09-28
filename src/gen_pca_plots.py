import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.pipeline import build_preprocess_pipeline, split_X_y

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "heart_disease.csv")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

df = pd.read_csv(DATA_PATH, na_values='?')
df['target'] = df['target'].apply(lambda x: 1 if float(x) > 0 else 0)
X, y = split_X_y(df)

pre = build_preprocess_pipeline(False)
X_tr = pre.fit_transform(X, y)

pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_tr)

# 1) Cumulative explained variance
cum = np.cumsum(pca.explained_variance_ratio_)
plt.figure()
plt.plot(range(1, len(cum)+1), cum, marker="o")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA — Cumulative explained variance")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(RESULTS, "pca_explained_variance.png"), dpi=150, bbox_inches="tight")
plt.close()

# 2) (Optional) Scatter PC1–PC2
if X_pca.shape[1] >= 2:
    plt.figure()
    plt.scatter(X_pca[:,0], X_pca[:,1], c=y, s=18, alpha=0.7)
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PCA — PC1 vs PC2")
    plt.savefig(os.path.join(RESULTS, "pca_scatter_pc1_pc2.png"), dpi=150, bbox_inches="tight")
    plt.close()

# 3) Store dataset PCA
pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
pca_df["target"] = y.values
pca_df.to_csv(os.path.join(RESULTS, "pca_transformed.csv"), index=False)

print("Saved: pca_explained_variance.png, pca_scatter_pc1_pc2.png, pca_transformed.csv")
