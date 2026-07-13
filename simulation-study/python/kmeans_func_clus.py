# this file performs k-means clustering for functional data
from scipy.integrate import trapezoid
import os
os.getcwd()
import numpy as np
import pandas as pd
import sys

class FunctionalKMeans:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _functional_distance(self, A, B, t):
        return np.sqrt(np.sum([trapezoid((A[i] - B[i]) ** 2, t) for i in range(A.shape[0])]))

    def _compute_centroid(self, cluster_members):
        # Simple element-wise mean (functional mean)
        return np.mean(cluster_members, axis=0)

    def fit(self, X, t):
        """
        X: ndarray of shape (p, n, m) -> p observations, each n functions at m time points
        t: array of shape (m,) -> time grid
        """
        rng = np.random.default_rng(self.random_state)
        p = X.shape[0]

        # Step 1: Initialize centroids randomly
        init_indices = rng.choice(p, self.n_clusters, replace=False)
        centroids = X[init_indices]

        for iteration in range(self.max_iter):
            # Step 2: Assign clusters
            labels = np.zeros(p, dtype=int)
            for i in range(p):
                dists = [self._functional_distance(X[i], centroids[c], t) for c in range(self.n_clusters)]
                labels[i] = np.argmin(dists)

            # Step 3: Update centroids
            new_centroids = np.zeros_like(centroids)
            for c in range(self.n_clusters):
                members = X[labels == c]
                if len(members) > 0:
                    new_centroids[c] = self._compute_centroid(members)
                else:
                    # Handle empty cluster by reinitializing randomly
                    new_centroids[c] = X[rng.choice(p)]

            # Step 4: Check convergence
            shift = sum(self._functional_distance(centroids[c], new_centroids[c], t) for c in range(self.n_clusters))
            if shift < self.tol:
                break
            centroids = new_centroids

        self.labels_ = labels
        self.cluster_centers_ = centroids
        return self

# reading the simulated data
seed_number = int(sys.argv[1])
sim_name = "../data/simulation_values_%d.csv"%seed_number
df_loaded = pd.read_csv(sim_name)

# Infer the shape from the max indices
p = df_loaded['i'].max() + 1
n = df_loaded['j'].max() + 1
m = df_loaded['k'].max() + 1

# Create an empty array and fill it
X = np.zeros((p, n, m), dtype=float)

for _, row in df_loaded.iterrows():
    i, j, k, val = int(row['i']), int(row['j']), int(row['k']), row['value']
    X[i, j, k] = val
t = np.linspace(0, 1, m)

#  Perform functional k-means clustering
model = FunctionalKMeans(n_clusters=3, random_state=0)
model.fit(X, t)

labels = model.labels_ + 1
output_name = "../results/kmeans_clus_%d.txt"%seed_number
with open(output_name, 'w') as file:
    for item in labels:
        file.write(f"{item}\n")
