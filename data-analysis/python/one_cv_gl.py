import math
import os
import sys
import numpy as np
from tqdm import tqdm
from skfda import FDataBasis, FDataGrid
from skfda.representation.basis import BSplineBasis, TensorBasis
import matplotlib.pyplot as plt

# make the shared `common/` package (containing estpure.py) importable
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "common"))
import estpure

# ---------------------------------------------------------------
# 1. Helper functions for computing operator norms and the off-diagonal norm
# ---------------------------------------------------------------
def tensor_to_operator_norm(T):
    """T: (n_I, n_I, m, m) -> (n_I, n_I) matrix of operator norms.
    Divides by (m-1) to approximate the L^2 operator norm."""
    n_I = T.shape[0]
    m   = T.shape[2]
    C   = np.zeros((n_I, n_I))
    for i in range(n_I):
        for j in range(n_I):
            ev, _ = np.linalg.eigh(T[i, j])
            C[i, j] = np.max(np.abs(ev))
    return C / (m - 1)


def off_diag_norm_tensor(T):
    """
    Compute ||T||_{off-diag} = sqrt(sum_{i != j} ||T_{ij}||^2_op)
    where ||T_{ij}||_op is the operator norm of the (m,m) block T[i,j],
    with the 1/(m-1) correction for L^2 operator norm.

    T : (n_I, n_I, m, m)
    """
    n_I = T.shape[0]
    m   = T.shape[2]
    off_diag_sq = 0.0
    for i in range(n_I):
        for j in range(n_I):
            if i != j:
                ev, _    = np.linalg.eigh(T[i, j])
                op_norm  = np.max(np.abs(ev)) / (m - 1)
                off_diag_sq += op_norm**2
    return math.sqrt(off_diag_sq)

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
# 5. Cross-validation
# ---------------------------------------------------------------
def _single_cv(X, n_basis, verbose = False):
    """One round of cross-validation. Returns optimal delta."""
    p, n, m = X.shape
    perm   = np.random.permutation(n)
    X_hold = X[:, perm[:n//2], :].copy()
    X2     = X[:, perm[n//2:], :].copy()

    X_hold -= X_hold.mean(axis=1, keepdims=True)
    X2     -= X2.mean(axis=1, keepdims=True)

    # Use condensed version directly -- never allocates (p,p,m,m)
    print("  CV fold: computing Sigma_hold condensed...")
    Sigma_hold_cond = compute_sigma_condensed(X_hold, n_basis=n_basis)
    print("  CV fold: computing Sigma2 condensed...")
    Sigma2_cond     = compute_sigma_condensed(X2,     n_basis=n_basis)

    #grid_deltas = (math.sqrt(math.log(p) / n) * np.arange(0.02, 0.50, 0.02) *
    #               np.quantile(Sigma2_cond.flatten(), 0.25))
    grid_deltas = [1, 10, 50, 100, 200, 400, 800, 1000, 1500]
    losses      = np.full(len(grid_deltas), np.inf)

    for l, delta in enumerate(grid_deltas):
        I, K = estpure.pure_var(Sigma2_cond, delta)
        if K == 1:
            losses[l] = 1.0
            continue

        I_flat = [x for Ia in I for x in Ia]
        A      = np.zeros((p, K))
        for a in range(K):
            I_a = sorted(I[a])
            A[I_a[0], a] = 1.0
            for j in I_a[1:]:
                A[j, a] = estpure.sign(Sigma2_cond[I_a[0], j])

        A_I    = A[I_flat, :]
        C_cond = np.zeros((K, K))
        for a in range(K):
            for b in range(K):
                if a == b:
                    pairs = [(i1, i2) for i1 in I[a]
                                      for i2 in I[a] if i1 != i2]
                    if len(I[a]) > 1:
                        C_cond[a, a] = (
                            sum(A[i1, a] * Sigma2_cond[i1, i2] * A[i2, a]
                                for i1, i2 in pairs)
                            / (len(I[a]) * (len(I[a]) - 1))
                        )
                    else:
                        C_cond[a, a] = (
                            sum(A[i1, a] * Sigma2_cond[i1, i2] * A[i2, a]
                                for i1, i2 in pairs)) # divide by 1
                else:
                    C_cond[a, b] = (
                        sum(A[k1, a] * Sigma2_cond[k1, k2] * A[k2, b]
                            for k1 in I[a] for k2 in I[b])
                        / (len(I[a]) * len(I[b]))
                    )

        W         = A_I @ C_cond @ A_I.T
        S_II      = Sigma_hold_cond[np.ix_(I_flat, I_flat)]
        losses[l] = (estpure.off_diag_norm(W, S_II)
                     / math.sqrt(max(len(I_flat) * (len(I_flat) - 1), 1)))

    if verbose:
        plt.plot(grid_deltas, losses)
        plt.show()
    return grid_deltas[np.argmin(losses)]

# ---------------------------------------------------------------
# 5. New Cross-validation
# ---------------------------------------------------------------
def _single_cv_new(X, n_basis, verbose=False):
    """
    Cross-validation following the paper exactly.

    First half  (X_hold) -> Sigma^1, used only for loss evaluation.
    Second half (X2)     -> Sigma^2, used for pure variable selection.

    Loss: (1/sqrt(|I|(|I|-1))) * ||Sigma^1_{II} - V_hat(q)||_{off-diag}
    where the off-diag norm sums operator norms of off-diagonal blocks.
    """
    p, n, m = X.shape
    perm   = np.random.permutation(n)
    X_hold = X[:, perm[:n//2], :].copy()
    X2     = X[:, perm[n//2:], :].copy()

    X_hold -= X_hold.mean(axis=1, keepdims=True)
    X2     -= X2.mean(axis=1, keepdims=True)

    # Condensed Sigma^2 for pure variable selection only
    print("  CV fold: computing Sigma2 condensed (for pure var selection)...")
    Sigma2_cond = compute_sigma_condensed(X2, n_basis=n_basis)

    # Data-driven delta grid
    grid_deltas = (np.quantile(Sigma2_cond.flatten(), 0.25) *
                   math.sqrt(math.log(p) / n) * np.arange(0.1, 5.0, 0.1))
    losses = np.full(len(grid_deltas), np.inf)

    # Precompute basis projections for X_hold (all parcels, needed for Sigma^1_II)
    print("  Projecting X_hold parcels to basis...")
    grid_points = np.arange(0, 1, 1/m)
    basis       = BSplineBasis(domain_range=(0, 1), n_basis=n_basis)
    basis_bivar = TensorBasis([basis, basis])
    eval_points = np.column_stack([
        g.ravel() for g in np.meshgrid(grid_points, grid_points, indexing="ij")
    ])
    coefs_hold = {}
    for i in tqdm(range(p)):
        coefs_hold[i] = project_to_basis(X_hold[i], grid_points, basis)

    # Also precompute X2 projections for C_hat computation
    print("  Projecting X2 parcels to basis...")
    coefs_X2 = {}
    for i in tqdm(range(p)):
        coefs_X2[i] = project_to_basis(X2[i], grid_points, basis)

    def _compute_block(coefs, i, j):
        """Compute (m,m) smoothed cross-cov block between parcels i and j."""
        n_obs = coefs[i].shape[0]
        cross_cov = (coefs[i].T @ coefs[j]) / (n_obs - 1)
        cov_tensor = FDataBasis(
            basis=basis_bivar,
            coefficients=cross_cov.flatten()[None, :]
        )
        return cov_tensor(eval_points).reshape(m, m) / (m - 1)

    losses = np.full(len(grid_deltas), np.inf)
    K_list = np.full(len(grid_deltas), -1)

    for l, delta in enumerate(grid_deltas):
        I, K = estpure.pure_var(Sigma2_cond, delta)
        K_list[l] = K
        if K <= 1:
            continue

        I_flat = [x for Ia in I for x in Ia]
        n_I    = len(I_flat)
        if n_I < 2:
            continue

        # --- A_I from Sigma2_cond ---
        A = np.zeros((p, K))
        for a in range(K):
            I_a = sorted(I[a])
            A[I_a[0], a] = 1.0
            for j in I_a[1:]:
                A[j, a] = estpure.sign(Sigma2_cond[I_a[0], j])
        A_I = A[I_flat, :]   # (n_I, K)

        # --- C_hat: (K, K, m, m) from X2 ---
        C_hat = np.zeros((K, K, m, m))
        for a in range(K):
            for b in range(K):
                if a == b:
                    Ia   = list(I[a])
                    norm = len(Ia) * (len(Ia) - 1) if len(Ia) > 1 else 1
                    for i in Ia:
                        for j in Ia:
                            if i != j:
                                C_hat[a, a] += (A[i, a] * A[j, a]
                                                * _compute_block(coefs_X2, i, j))
                    C_hat[a, a] /= norm
                else:
                    Ia, Ib = list(I[a]), list(I[b])
                    norm   = len(Ia) * len(Ib)
                    for i in Ia:
                        for j in Ib:
                            C_hat[a, b] += (A[i, a] * A[j, b]
                                            * _compute_block(coefs_X2, i, j))
                    C_hat[a, b] /= norm

        # --- V_hat = A_I C_hat A_I^T: (n_I, n_I, m, m) block operator ---
        # V_hat[ii, jj] = sum_{a,b} A_I[ii,a] * C_hat[a,b] * A_I[jj,b]
        V_hat = np.zeros((n_I, n_I, m, m))
        for ii in range(n_I):
            for jj in range(n_I):
                for a in range(K):
                    for b in range(K):
                        V_hat[ii, jj] += (A_I[ii, a] * A_I[jj, b]
                                          * C_hat[a, b])

        # --- Sigma^1_{II}: (n_I, n_I, m, m) from X_hold ---
        Sigma1_II = np.zeros((n_I, n_I, m, m))
        for ii, i in enumerate(I_flat):
            for jj, j in enumerate(I_flat):
                if ii <= jj:
                    block = _compute_block(coefs_hold, i, j)
                    Sigma1_II[ii, jj] = block
                    Sigma1_II[jj, ii] = block.T

        # --- Off-diagonal norm of (Sigma^1_II - V_hat) ---
        # ||Psi||_{off-diag} = sqrt(sum_{i != j} ||Psi_{ij}||^2_op)
        diff = Sigma1_II - V_hat  # (n_I, n_I, m, m)
        losses[l] = off_diag_norm_tensor(diff) / math.sqrt(n_I * (n_I - 1))

    print(f"  K values: min={K_list.min()}, max={K_list.max()}")
    print(f"  Losses: {losses[np.isfinite(losses)]}")
    print(f"grid deltas: {grid_deltas}")

    if verbose:
        finite = np.isfinite(losses)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        ax1.plot(grid_deltas[finite], losses[finite], marker='o')
        ax1.set_xlabel("delta"); ax1.set_ylabel("CV loss")
        ax2.plot(grid_deltas, K_list, marker='o')
        ax2.set_xlabel("delta"); ax2.set_ylabel("K")
        plt.tight_layout(); plt.show()

    return grid_deltas[np.argmin(losses)]

# ---------------------------------------------------------------
# 9. Entry point
# ---------------------------------------------------------------
if __name__ == "__main__":
    # input the seed
    seed = int(sys.argv[1])
    np.random.seed(seed)

    X = np.load("../data/aomic_data_200.npy")
    Xc = X - X.mean(axis=1, keepdims=True)
    delta_from_cv = _single_cv_new(Xc, n_basis=10, verbose=False)
    os.makedirs("../results", exist_ok=True)
    np.savetxt(f"../results/delta_{seed}.txt", [delta_from_cv])
