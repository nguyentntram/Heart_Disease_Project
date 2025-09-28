import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from src.pipeline import build_preprocess_pipeline, split_X_y
from src.unsupervised import kmeans_elbow, run_kmeans, run_agglomerative, compare_to_labels

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "heart_disease.csv")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

df = pd.read_csv(DATA_PATH, na_values='?')
df['target'] = df['target'].apply(lambda x: 1 if float(x) > 0 else 0)
X, y = split_X_y(df)

pre = build_preprocess_pipeline(False)
X_tr = pre.fit_transform(X, y)

# --- KMeans elbow ---
elbow = kmeans_elbow(X_tr, 2, 8)
plt.figure()
plt.plot(elbow["k"], elbow["inertia"], marker="o")
plt.xlabel("k"); plt.ylabel("Inertia"); plt.title("KMeans — Elbow")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(RESULTS, "kmeans_elbow.png"), dpi=150, bbox_inches="tight")
plt.close()

# Chọn k=2 (thường hợp lý với nhãn nhị phân)
km = run_kmeans(X_tr, k=2)
ari_km = compare_to_labels(km["labels"], y.to_numpy())

# --- Hierarchical + Dendrogram ---
Z = linkage(X_tr[:200], method="ward")  # cắt 200 điểm cho đồ thị gọn
plt.figure(figsize=(9, 4))
dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
plt.title("Hierarchical Clustering — Dendrogram (subset)")
plt.savefig(os.path.join(RESULTS, "hierarchical_dendrogram.png"), dpi=150, bbox_inches="tight")
plt.close()

ag = run_agglomerative(X_tr, n_clusters=2, linkage_method="ward")
ari_ag = compare_to_labels(ag["labels"], y.to_numpy())

with open(os.path.join(RESULTS, "unsupervised_metrics.txt"), "w", encoding="utf-8") as f:
    f.write(f"Silhouette (KMeans, k=2): {km['silhouette']}\n")
    f.write(f"ARI (KMeans vs labels): {ari_km}\n")
    f.write(f"Silhouette (Agglomerative, 2 clusters): {ag['silhouette']}\n")
    f.write(f"ARI (Agglomerative vs labels): {ari_ag}\n")

print("Saved: kmeans_elbow.png, hierarchical_dendrogram.png, unsupervised_metrics.txt")
