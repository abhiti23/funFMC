import os
import sys
import math
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
# from joblib import Parallel, delayed

from skfda import FDataBasis, FDataGrid
from skfda.representation.basis import BSplineBasis, TensorBasis
import matplotlib.pyplot as plt
import seaborn as sns

# make the shared `common/` package (containing estpure.py) importable
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "common"))
import estpure

# ---------------------------------------------------------------
# 2. Covariance estimation  (key speedup: project basis once per parcel)
# ---------------------------------------------------------------
def project_to_basis(X_i, grid_points, basis):
    """
    Project one parcel's timeseries (n x m) onto B-spline basis.
    Returns centered coefficient matrix of shape (n, n_basis).
    """
    fd       = FDataGrid(data_matrix=X_i, grid_points=grid_points)
    fd_basis = fd.to_basis(basis)
    coef     = fd_basis.coefficients                        # (n, n_basis)
    return coef - coef.mean(axis=0)                         # center


def compute_sigma_hat(X, n_basis=10, n_jobs=1):
    """
    Compute the smoothed cross-covariance tensor Sigma_hat of shape (p, p, m, m).

    Speedup over original: B-spline basis is fitted once per parcel (p fits),
    then all p^2 cross-covariances are computed from coefficient matrices.
    Original code called cov_fda p^2 times, fitting the basis twice each call.

    Parameters
    ----------
    X       : (p, n, m) array
    n_basis : number of B-spline basis functions
    n_jobs  : number of parallel workers (-1 = all cores)

    Returns
    -------
    Sigma_hat : (p, p, m, m) array
    """
    p, n, m       = X.shape
    grid_points   = np.arange(0, 1, 1 / m)
    basis         = BSplineBasis(domain_range=(0, 1), n_basis=n_basis)
    basis_bivar   = TensorBasis([basis, basis])
    eval_points   = np.column_stack([
        g.ravel() for g in np.meshgrid(grid_points, grid_points, indexing="ij")
    ])

    # --- Step 1: project each parcel once ---
    print("  Projecting parcels to B-spline basis...")
    coefs = []   # list of (n, n_basis) arrays, one per parcel
    for i in tqdm(range(p)):
        coefs.append(project_to_basis(X[i], grid_points, basis))  # (n, n_basis)

    # --- Step 2: compute all (i,j) cross-covariances from coefficients ---
    # cross_cov[i,j] = coefs[i].T @ coefs[j] / (n-1)  shape (n_basis, n_basis)
    # then evaluate on grid to get (m, m) matrix
    print("  Computing cross-covariance matrices...")

    def _cov_ij(i, j):
        cross_cov    = (coefs[i].T @ coefs[j]) / (n - 1)   # (n_basis, n_basis)
        cov_tensor   = FDataBasis(
            basis=basis_bivar,
            coefficients=cross_cov.flatten()[None, :]
        )
        return cov_tensor(eval_points).reshape(m, m)

    # Compute upper triangle in parallel, then mirror
    pairs = [(i, j) for i in range(p) for j in range(i, p)]

    results = [_cov_ij(i, j) for i, j in tqdm(pairs)]

    Sigma_hat = np.zeros((p, p, m, m), dtype=float)
    for (i, j), cov_ij in zip(pairs, results):
        Sigma_hat[i, j] = cov_ij
        Sigma_hat[j, i] = cov_ij.T   # symmetry: Sigma_{ji}(s,t) = Sigma_{ij}(t,s)

    return Sigma_hat

# since computing Sigma_hat is too memory-intense, we will store only
# subblocks at one given time
def compute_sigma_subblocks(X, I_flat, n_basis=10):
    """
    Compute only the subblocks of Sigma_hat needed for make_AI and make_AJ.

    Returns
    -------
    Sigma_II  : (|I|, |I|, m, m)  -- needed for make_AI and C
    Sigma_IJ  : (|I|, n_J, m, m)  -- needed for U in make_AJ
    """
    p, n, m = X.shape
    list_J = sorted(set(range(p)) - set(I_flat))
    n_I = len(I_flat)
    n_J = len(list_J)
    grid_points = np.arange(0, 1, 1 / m)
    basis = BSplineBasis(domain_range=(0, 1), n_basis=n_basis)
    basis_bivar = TensorBasis([basis, basis])
    eval_points = np.column_stack([
        g.ravel() for g in np.meshgrid(grid_points, grid_points, indexing="ij")
    ])

    # Project only the parcels we need: I_flat and J
    all_indices = I_flat + list_J  # |I| + n_J indices
    print("  Projecting relevant parcels to basis...")
    coefs = {}
    for idx in tqdm(all_indices):
        coefs[idx] = project_to_basis(X[idx], grid_points, basis)

    def _block(i, j):
        cross_cov = (coefs[i].T @ coefs[j]) / (n - 1)
        cov_tensor = FDataBasis(
            basis=basis_bivar,
            coefficients=cross_cov.flatten()[None, :]
        )
        return cov_tensor(eval_points).reshape(m, m)

    # Sigma_II
    print("  Computing Sigma_II...")
    Sigma_II = np.zeros((n_I, n_I, m, m))
    for ii, i in enumerate(tqdm(I_flat)):
        for jj, j in enumerate(I_flat):
            if ii <= jj:
                block = _block(i, j)
                Sigma_II[ii, jj] = block
                Sigma_II[jj, ii] = block.T

    # Sigma_IJ
    print("  Computing Sigma_IJ...")
    Sigma_IJ = np.zeros((n_I, n_J, m, m))
    for ii, i in enumerate(tqdm(I_flat)):
        for jj, j in enumerate(list_J):
            Sigma_IJ[ii, jj] = _block(i, j)

    return Sigma_II, Sigma_IJ, list_J


# ---------------------------------------------------------------
# 3. Tensor condensation
# ---------------------------------------------------------------

def tensor_to_operator_norm(T):
    """T: (p, p, m, m) -> (p, p) matrix of operator norms."""
    p = T.shape[0]
    m = T.shape[2]
    C = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            ev, _ = np.linalg.eigh(T[i, j])
            C[i, j] = np.max(np.abs(ev))
    return C/(m-1)

def compute_sigma_condensed(X, n_basis=10):
    """
    Compute operator-norm condensed Sigma directly, without storing
    the full (p, p, m, m) tensor. Memory: O(p^2) instead of O(p^2 m^2).
    """
    p, n, m     = X.shape
    grid_points = np.arange(0, 1, 1/m)
    basis       = BSplineBasis(domain_range=(0, 1), n_basis=n_basis)
    basis_bivar = TensorBasis([basis, basis])
    eval_points = np.column_stack([
        g.ravel() for g in np.meshgrid(grid_points, grid_points, indexing="ij")
    ])

    # Project each parcel once
    print("  Projecting parcels to B-spline basis...")
    coefs = []
    for i in tqdm(range(p)):
        coefs.append(project_to_basis(X[i], grid_points, basis))

    # Compute condensed matrix directly, never storing full tensor
    print("  Computing condensed covariance matrix...")
    Sigma_cond = np.zeros((p, p))
    for i in tqdm(range(p)):
        for j in range(i, p):
            cross_cov  = (coefs[i].T @ coefs[j]) / (n - 1)   # (n_basis, n_basis)
            cov_tensor = FDataBasis(
                basis=basis_bivar,
                coefficients=cross_cov.flatten()[None, :]
            )
            block = cov_tensor(eval_points).reshape(m, m)     # (m, m)
            ev, _ = np.linalg.eigh(block)
            op_norm = np.max(np.abs(ev))
            Sigma_cond[i, j] = op_norm
            Sigma_cond[j, i] = op_norm   # symmetric

    return Sigma_cond/(m-1)

# ---------------------------------------------------------------
# 4. A_I and A_J estimation (unchanged logic, minor cleanup)
# ---------------------------------------------------------------
def make_AI(p, I, K, Sigma_hat):
    """
    Construct A_I from the pure variable partition I and Sigma_hat.
    Sigma_hat : (p, p, m, m)
    Returns AI : (|I|, K)
    """
    A      = np.zeros((p, K))
    list_I = [i for Ia in I for i in Ia]

    for a in range(K):
        list_Ia    = list(I[a])
        A[list_Ia[0], a] = 1.0
        ref = list_Ia[0]
        for j in list_Ia[1:]:
            S_add = Sigma_hat[ref, ref] + Sigma_hat[ref, j]   # (m, m)
            S_sub = Sigma_hat[ref, ref] - Sigma_hat[ref, j]
            ev_add, _ = np.linalg.eigh(S_add)
            ev_sub, _ = np.linalg.eigh(S_sub)
            A[j, a]   = 1.0 if ev_add.max() > ev_sub.max() else -1.0

    return A[list_I, :]   # (|I|, K)


def make_AJ(Sigma_hat, AI, I, K):
    """
    Estimate A_J and C from Sigma_hat and A_I.
    Sigma_hat : (p, p, m, m)
    Returns AJ : (n_J, K),  C : (K, K, m, m)
    """
    p, _, m = Sigma_hat.shape[0], Sigma_hat.shape[1], Sigma_hat.shape[-1]
    list_I  = [i for Ia in I for i in Ia]
    list_J  = sorted(set(range(p)) - set(list_I))
    lengthI = len(list_I)
    n_J     = len(list_J)

    # Full A with A_I filled in
    A = np.zeros((p, K))
    for idx, global_i in enumerate(list_I):
        A[global_i, :] = AI[idx, :]

    # --- C : (K, K, m, m) ---
    C = np.zeros((K, K, m, m))
    for a in range(K):
        for b in range(K):
            if a == b:
                Ia   = list(I[a])
                norm = len(Ia) * (len(Ia) - 1)
                for i in Ia:
                    for j in Ia:
                        if i != j:
                            C[a, a] += A[i, a] * A[j, a] * Sigma_hat[i, j]
                C[a, a] /= norm
            else:
                Ia, Ib = list(I[a]), list(I[b])
                norm   = len(Ia) * len(Ib)
                for i in Ia:
                    for j in Ib:
                        C[a, b] += A[i, a] * A[j, b] * Sigma_hat[i, j]
                C[a, b] /= norm

    # --- V = InverseMatrix : (K, K) ---
    # V_{s,s2} = sum_k trace(C[k,s2]^T C[k,s])
    V_mat = np.zeros((K, K))
    for s in range(K):
        for s2 in range(K):
            V_mat[s, s2] = sum(
                np.trace(C[k, s2].T @ C[k, s]) for k in range(K)
            )

    # --- U : (K, n_J, m, m) ---
    W       = AI.T @ AI                          # (K, K)
    WinvAIT = np.linalg.inv(W) @ AI.T           # (K, |I|)

    U = np.zeros((K, n_J, m, m))
    for k in range(K):
        for jidx, j in enumerate(list_J):
            for ii, i_global in enumerate(list_I):
                U[k, jidx] += WinvAIT[k, ii] * Sigma_hat[i_global, j]

    # --- CU : (K, n_J) ---
    CU = np.zeros((K, n_J))
    for s in range(K):
        for jidx in range(n_J):
            CU[s, jidx] = sum(
                np.trace(U[k, jidx].T @ C[k, s]) for k in range(K)
            )

    try:
        AJ = np.linalg.solve(V_mat, CU)         # (K, n_J)
    except np.linalg.LinAlgError:
        print("  Warning: V matrix singular, using least-squares fallback.")
        AJ, _, _, _ = np.linalg.lstsq(V_mat, CU, rcond=None)

    return AJ.T, C   # (n_J, K), (K, K, m, m)

# Versions of the above functions that only require one block at a time
def make_AI_sub(I, K, Sigma_II, I_flat):
    """
    Same as make_AI but takes Sigma_II (|I| x |I| x m x m)
    with rows/cols indexed by position in I_flat.
    """
    n_I = len(I_flat)
    idx_map = {global_i: ii for ii, global_i in enumerate(I_flat)}
    A = np.zeros((max(I_flat)+1, K))   # only needs rows in I_flat

    for a in range(K):
        list_Ia = list(I[a])
        ref     = list_Ia[0]
        ii_ref  = idx_map[ref]
        A[ref, a] = 1.0
        for j in list_Ia[1:]:
            ii_j  = idx_map[j]
            S_add = Sigma_II[ii_ref, ii_ref] + Sigma_II[ii_ref, ii_j]
            S_sub = Sigma_II[ii_ref, ii_ref] - Sigma_II[ii_ref, ii_j]
            ev_add, _ = np.linalg.eigh(S_add)
            ev_sub, _ = np.linalg.eigh(S_sub)
            A[j, a]   = 1.0 if ev_add.max() > ev_sub.max() else -1.0

    AI = np.array([A[i, :] for i in I_flat])
    return AI


def make_AJ_sub(Sigma_II, Sigma_IJ, AI, I, K, I_flat, list_J):
    """
    Same as make_AJ but takes subblocks instead of full Sigma_hat.
    Sigma_II : (|I|, |I|, m, m)
    Sigma_IJ : (|I|, n_J, m, m)
    """
    m   = Sigma_II.shape[-1]
    n_I = len(I_flat)
    n_J = len(list_J)
    idx_map = {global_i: ii for ii, global_i in enumerate(I_flat)}

    # A weights for pure variables
    A_weights = {}
    for a in range(K):
        for ii, i in enumerate(I_flat):
            A_weights[i] = AI[ii, :]

    # C : (K, K, m, m) -- uses only Sigma_II
    C = np.zeros((K, K, m, m))
    for a in range(K):
        for b in range(K):
            if a == b:
                Ia   = list(I[a])
                if len(Ia) == 1:
                    norm = 1
                else:
                    norm = len(Ia) * (len(Ia) - 1)
                for i in Ia:
                    for j in Ia:
                        if i != j:
                            ii, jj = idx_map[i], idx_map[j]
                            C[a, a] += A_weights[i][a] * A_weights[j][a] * Sigma_II[ii, jj]
                C[a, a] /= norm
            else:
                Ia, Ib = list(I[a]), list(I[b])
                norm   = len(Ia) * len(Ib)
                for i in Ia:
                    for j in Ib:
                        ii, jj = idx_map[i], idx_map[j]
                        C[a, b] += A_weights[i][a] * A_weights[j][b] * Sigma_II[ii, jj]
                C[a, b] /= norm

    # V : (K, K)
    V_mat = np.zeros((K, K))
    for s in range(K):
        for s2 in range(K):
            V_mat[s, s2] = sum(np.trace(C[k, s2].T @ C[k, s])/(m-1)**2 for k
                               in
                               range(K))

    # U : (K, n_J, m, m) -- uses Sigma_IJ
    W       = AI.T @ AI
    WinvAIT = np.linalg.inv(W) @ AI.T   # (K, |I|)

    U = np.zeros((K, n_J, m, m))
    for k in range(K):
        for jidx in range(n_J):
            for ii in range(n_I):
                U[k, jidx] += WinvAIT[k, ii] * Sigma_IJ[ii, jidx]

    # CU : (K, n_J)
    CU = np.zeros((K, n_J))
    for s in range(K):
        for jidx in range(n_J):
            CU[s, jidx] = sum(np.trace(U[k, jidx].T @ C[k, s])/(m-1)**2 for k
                              in
                              range(K))

    try:
        AJ = np.linalg.solve(V_mat, CU)
    except np.linalg.LinAlgError:
        AJ, _, _, _ = np.linalg.lstsq(V_mat, CU, rcond=None)

    return AJ.T, C

# ---------------------------------------------------------------
# 6. Post-processing: sparsify A_J
# ---------------------------------------------------------------
def sparsify_AJ(AJ, upper=0.90, lower=0.05):
    """
    Threshold A_J: entries > upper -> ±1 (and zero out other columns),
                   entries < lower -> 0.
    """
    AJ = AJ.copy()
    for row in range(AJ.shape[0]):
        for col in range(AJ.shape[1]):
            if abs(AJ[row, col]) > upper:
                AJ[row, col] = math.copysign(1.0, AJ[row, col])
                other = [c for c in range(AJ.shape[1]) if c != col]
                AJ[row, other] = 0.0
    for row in range(AJ.shape[0]):
        for col in range(AJ.shape[1]):
            if abs(AJ[row, col]) < lower:
                AJ[row, col] = 0.0
    return AJ


# ---------------------------------------------------------------
# 7. Functions for figures
# ---------------------------------------------------------------

def make_heatmap(Sigma_cond):
    """This functions takes in a matrix of dimension (p,p) and makes  a
    heatmap. In the context of our algorithm, we plot Sigma_condensed to
    assess if we can visually identify clusters. We also save the numerical
    value of Sigma_condensed as a csv."""
    sns.heatmap(Sigma_cond, cmap="YlGnBu")
    # plt.ylim(0, 212)
    plt.savefig(f"{OUT_DIR}/Sigma_condensed_aomic.png")
    pd.DataFrame(Sigma_cond).to_csv(f"{OUT_DIR}/Sigma_condensed_aomic.csv",
                                    index=False)


if __name__ == "__main__":
    # Single results directory shared by the SLURM CV step (one_cv_gl.py),
    # this script, and the plotting scripts, so everything reads/writes the
    # same place whether it was produced on the cluster or run locally.
    OUT_DIR = "../results"
    IN_DIR = "../results"
    n_basis = 10

    # all_deltas.txt is produced by concatenating the per-seed CV outputs,
    # e.g.: cat ../results/delta_*.txt > ../results/all_deltas.txt
    deltas = np.loadtxt(f"{IN_DIR}/all_deltas.txt")
    opt_delta = statistics.median(deltas)
    print(f"Optimal delta (median of {len(np.atleast_1d(deltas))} CVs): {opt_delta:.4f}")

    # Step 2: compute condensed Sigma for pure variable selection
    X = np.load("../data/aomic_data_200.npy")
    p, n, m = X.shape
    Xc = X - X.mean(axis=1, keepdims=True)

    # Step 0: compute (and cache) the condensed Sigma_hat used for pure
    # variable selection. This used to load a stale copy from a different,
    # hand-managed results folder left over from local vs. cluster runs;
    # now it's computed once and cached under the same OUT_DIR everything
    # else uses, so a fresh clone of the repo can reproduce it end to end.
    sigma_cond_path = f"{OUT_DIR}/Sigma_condensed_aomic.csv"
    if os.path.exists(sigma_cond_path):
        print(f"\nLoading cached condensed Sigma_hat from {sigma_cond_path}...")
        Sigma_cond = pd.read_csv(sigma_cond_path).to_numpy()
    else:
        print("\nComputing condensed Sigma_hat (no cache found)...")
        Sigma_cond = compute_sigma_condensed(Xc, n_basis=n_basis)
        make_heatmap(Sigma_cond)

    print("\nSelecting pure variables...")
    est_I, est_K = estpure.pure_var(Sigma_cond, opt_delta)
    I_flat = [i for Ia in est_I for i in Ia]
    print(f"Estimated K = {est_K}")

    # The next chunk is added to check if we will run into memory issues
    # while computing Sigma_II and Sigma_IJ. The hope is that I is small,
    # so neither will be a problem.
    n_I = len(I_flat)
    n_J = p - n_I
    mem_II = n_I * n_I * m * m * 8 / 1e9
    mem_IJ = n_I * n_J * m * m * 8 / 1e9
    print(
        f"  Memory estimate: Sigma_II={mem_II:.2f} GB, Sigma_IJ={mem_IJ:.2f} GB")

    # Step 3: compute only the subblocks needed
    print("\nComputing Sigma subblocks...")
    Sigma_II, Sigma_IJ, list_J = compute_sigma_subblocks(Xc, I_flat,
                                                         n_basis=n_basis)

    # Step 4: estimate A
    est_AI = make_AI_sub(est_I, est_K, Sigma_II, I_flat)
    est_AJ_raw, est_C = make_AJ_sub(Sigma_II, Sigma_IJ, est_AI,
                                    est_I, est_K, I_flat, list_J)
    est_AJ = sparsify_AJ(est_AJ_raw)

    # Assemble full A
    est_A = np.zeros((p, est_K))
    for idx, i in enumerate(I_flat):
        est_A[i, :] = est_AI[idx, :]
    for idx, j in enumerate(list_J):
        est_A[j, :] = est_AJ[idx, :]

    # saving results
    pd.DataFrame(est_A).to_csv(f"{OUT_DIR}/est_A.csv", index=False)
    with open(f"{OUT_DIR}/est_K.txt", "w") as f:
        f.write(str(est_K) + "\n")

    I_flat = [i for Ia in est_I for i in Ia]
    pd.DataFrame({
        "parcel": I_flat,
        "cluster": [a for a, Ia in enumerate(est_I) for _ in Ia]
    }).to_csv(f"{OUT_DIR}/est_I.csv", index=False)

    print(f"\n=== Results ===")
    print(f"  Estimated number of clusters K = {est_K}")
    print(f"  est_A  saved to {OUT_DIR}/est_A.csv")
    print(f"  est_I  saved to {OUT_DIR}/est_I.csv")
    print(f"  est_K  saved to {OUT_DIR}/est_K.txt")