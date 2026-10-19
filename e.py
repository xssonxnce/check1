# ==========================================
# 0. ALL IMPORTS (Keep these at the top)
# ==========================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch

# ==========================================
# 0B. DATA LOADING & SETUP
# ==========================================
# >>> EXAM STEP: Replace 'X' below with whatever variable your professor 
# provided in the exam's starting code (e.g., X = X_train, X = dataset, etc.)
X = data.data  # <-- CHANGE THIS TO MATCH YOUR EXAM'S DATA VARIABLE

# Standardize features (Crucial for PCA, K-Means, and GMM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Data loaded and scaled successfully! Shape: {X.shape}\n")


# ==========================================
# BLOCK 1: PRINCIPAL COMPONENT ANALYSIS (PCA)
# (Delete or comment out if PCA is NOT on your exam)
# ==========================================
print("--- Running PCA ---")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

eigenvalues = pca.explained_variance_
eigenvectors = pca.components_
exp_var_ratio = pca.explained_variance_ratio_

print("Eigenvalues:", eigenvalues)
print("Explained Variance Ratio:", exp_var_ratio * 100)
print(f"Total Variance Captured: {np.sum(exp_var_ratio) * 100:.2f}%\n")


# ==========================================
# BLOCK 2: K-MEANS CLUSTERING
# (Delete or comment out if K-Means is NOT on your exam)
# ==========================================
print("--- Running K-Means ---")
# Find optimal k using silhouette score
best_k = 2
best_score = -1
for k in range(2, 6):
    kmeans_temp = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels_temp = kmeans_temp.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels_temp)
    if score > best_score:
        best_score = score
        best_k = k

print(f"Optimal k selected: {best_k}")

# Train final model
kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

print("Cluster Centroids:\n", kmeans.cluster_centers_)
print("Inertia (WCSS):", kmeans.inertia_, "\n")


# ==========================================
# BLOCK 3: DBSCAN CLUSTERING
# (Delete or comment out if DBSCAN is NOT on your exam)
# ==========================================
print("--- Running DBSCAN ---")
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

n_clusters_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise_db = list(dbscan_labels).count(-1)

print(f"DBSCAN Clusters found: {n_clusters_db}")
print(f"DBSCAN Noise points found: {n_noise_db}\n")


# ==========================================
# BLOCK 4: HIERARCHICAL CLUSTERING
# (Delete or comment out if Hierarchical is NOT on your exam)
# ==========================================
print("--- Running Hierarchical Clustering ---")
# Optional Dendrogram plot (uncomment if you want to view it)
# plt.figure(figsize=(5, 3))
# sch.dendrogram(sch.linkage(X_scaled, method='ward'))
# plt.show()

agg = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
agg_labels = agg.fit_predict(X_scaled)
print("Hierarchical clustering completed successfully. Labels assigned.\n")


# ==========================================
# BLOCK 5: GAUSSIAN MIXTURE MODELS (GMM)
# (Delete or comment out if GMM is NOT on your exam)
# ==========================================
print("--- Running GMM ---")
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X_scaled)

gmm_labels = gmm.predict(X_scaled)
gmm_probs = gmm.predict_proba(X_scaled)

print(f"GMM BIC Score: {gmm.bic(X_scaled):.2f}")
print(f"GMM AIC Score: {gmm.aic(X_scaled):.2f}\n")
