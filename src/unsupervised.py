import numpy as np
from typing import Dict, Any, List
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

def kmeans_elbow(X: np.ndarray, k_min: int = 2, k_max: int = 10) -> Dict[str, Any]:
    inertias = []
    ks = list(range(k_min, k_max + 1))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        km.fit(X)
        inertias.append(km.inertia_)
    return {"k": ks, "inertia": inertias}

def run_kmeans(X: np.ndarray, k: int = 2) -> Dict[str, Any]:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels) if len(set(labels)) > 1 else None
    return {"model": km, "labels": labels, "silhouette": sil}

def run_agglomerative(X: np.ndarray, n_clusters: int = 2, linkage_method: str = "ward") -> Dict[str, Any]:
    ag = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = ag.fit_predict(X)
    sil = silhouette_score(X, labels) if len(set(labels)) > 1 else None
    return {"model": ag, "labels": labels, "silhouette": sil}

def compare_to_labels(pred_labels: np.ndarray, true_labels: np.ndarray) -> float:
    return adjusted_rand_score(true_labels, pred_labels)