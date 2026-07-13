# this file contains the code for clustering functional data using our proposed method based on factor models.
import os
import sys
os.getcwd()
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import statistics
from tqdm import tqdm
import random
from scipy.stats import norm

# make the shared `common/` package (containing estpure.py) importable
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "common"))
import estpure

# packages for FDA
from skfda import FDataBasis
from skfda import FDataGrid
from skfda.representation.basis import BSplineBasis
from skfda.representation.basis import TensorBasis
from skfda.misc.covariances import Matern
from skfda.datasets import make_gaussian_process
from itertools import combinations, permutations

# read in the seed number
seed_number = int(sys.argv[1])
# reading the simulated data
csv_name = '../data/simulation_values_%d.csv'%seed_number
df_loaded = pd.read_csv(csv_name)

# Infer the shape from the max indices
p = df_loaded['i'].max() + 1
n = df_loaded['j'].max() + 1
m = df_loaded['k'].max() + 1

# Create an empty array and fill it
X = np.zeros((p, n, m), dtype=float)

for _, row in df_loaded.iterrows():
    i, j, k, val = int(row['i']), int(row['j']), int(row['k']), row['value']
    X[i, j, k] = val

# helper function that takes in a tensor of covariance matrices and returns a matrix of trace norms
def tensor_to_trace_norm(T):
    # T has size K,K,m,m
    K = T.shape[0]
    C = np.zeros((K,K), float)
    for i in range(K):
        for j in range(K):
            C[i,j] = np.trace(T[i,j,:,:])
    return C

def tensor_to_operator_norm(T):
    # T has size K,K,m,m
    K = T.shape[0]
    C = np.zeros((K,K), float)
    for i in range(K):
        for j in range(K):
            ev, _ = np.linalg.eigh(T[i,j,:,:])
            C[i,j] = max(abs(ev))
    return C

# This function takes in n samples of two functions Xi and Xj (each m dimensional) and calculates the smoothed cross-covariance matrix using B-spline
# basis expansion for the 2D covariance surface. We assume that both functions are sampled on the same equidistant sequence of points on [0,1].
def cov_fda(Xi, Xj, n_basis=10):
    n, m = Xi.shape
    if Xi.shape != Xj.shape: return "Error: The two functional data have mismatching dimensions"
    grid_points = np.arange(0, 1, 1 / m)
    basis = BSplineBasis(domain_range=(0, 1), n_basis=n_basis)
    basis_bivar = TensorBasis([basis, basis, ])

    fd = FDataGrid(data_matrix=Xi, grid_points=grid_points)
    fd_basis = fd.to_basis(basis)

    fd1 = FDataGrid(data_matrix=Xj, grid_points=grid_points)
    fd1_basis = fd1.to_basis(basis)

    # Cross-covariance function between the first and eleventh X
    fd_centered = fd_basis.coefficients - np.mean(fd_basis.coefficients, axis=0)
    fd1_centered = fd1_basis.coefficients - np.mean(fd1_basis.coefficients, axis=0)

    # Compute cross-covariance matrix
    cross_cov = (fd_centered.T @ fd1_centered) / (fd_centered.shape[0] - 1)
    cov_matrix_tensor = FDataBasis(basis=basis_bivar, coefficients=cross_cov.flatten()[None, :])
    s_grid, t_grid = np.meshgrid(grid_points, grid_points, indexing="ij")
    points = np.column_stack([s_grid.ravel(), t_grid.ravel()])

    # Evaluate function
    cov_matrix_coeff = cov_matrix_tensor(points).reshape(m, m)
    return cov_matrix_coeff


# function to select delta using cross-validation with smoothed covariance using a B-spline basis.
def cross_val_mv_smooth(X, return_type="delta"):
    '''This function takes in a tensor X and applies the LOVE algorithm to find K. It returns the delta for which cv loss is the least'''
    # return_type can be "delta", which returns the optimal delta or "graph" which produces graphs of the CV loss against each delta

    p, n, m = X.shape
    # split X into two halves for cross-validation
    perm = np.arange(n)
    np.random.shuffle(perm)
    Xperm = X[:, perm, :]

    # we will divide X into two halves
    X_hold = Xperm[:, : (n // 2), :]
    X2 = Xperm[:, (n // 2):, :]
    # normalizing the new X's
    mean_X_hold = X_hold.mean(axis=1)
    for i in range(n // 2):
        X_hold[:, i, :] = X_hold[:, i, :] - mean_X_hold
    Sigma_hold_smooth = np.zeros((p, p, m, m), dtype=float)
    # finding the covariance tensor of Z's
    for i in range(p):
        for j in range(p):
            Sigma_hold_smooth[i, j, :, :] = cov_fda(X_hold[i, :, :], X_hold[j, :, :])
    Sigma_hold_condensed = tensor_to_operator_norm(Sigma_hold_smooth)

    mean_X2 = X2.mean(axis=1)
    for i in range(n // 2):
        X2[:, i, :] = X2[:, i, :] - mean_X2
    Sigma_smooth = np.zeros((p, p, m, m), dtype=float)
    for i in range(p):
        for j in range(p):
            Sigma_smooth[i, j, :, :] = cov_fda(X2[i, :, :], X2[j, :, :])
    Sigma_condensed = tensor_to_operator_norm(Sigma_smooth)

    grid_deltas = np.arange(10.0, 50.0, 0.5)  # this was defined using trial and error
    grid_deltas = math.sqrt(math.log(p) / n) * grid_deltas
    losses = np.full(len(grid_deltas), -1.0)
    l = 0
    K_list = np.full(len(grid_deltas), -1)
    for delta in grid_deltas:
        # constructing A_I
        I, K = estpure.pure_var(Sigma_condensed, delta)
        # print("Number of clusters for tuning parameter", delta, "is :",K)

        lengthI = sum([len(elt) for elt in I])
        A = np.full((p, K), 0.0)
        for a in range(K):
            I_a = sorted(I[a])
            i = I_a[0]
            A[i, a] = 1
            I_a.remove(I_a[0])
            for j in I_a:
                A[j, a] = 1 * estpure.sign(Sigma_condensed[i, j])
        # extracting A_I from A
        I_flat = [x for xs in I for x in xs]  # flatten I
        A_I = A[I_flat, :]

        # constructing C_tr
        C_condensed = np.full((K, K), 0.0)
        for a in range(K):
            for b in range(K):
                if a == b:
                    sum_sigma = 0
                    for i1 in I[a]:
                        for i2 in I[a]:
                            if i1 != i2:
                                sum_sigma += Sigma_condensed[i1, i2]
                    C_condensed[a, a] = sum_sigma / (len(I[a]) * (len(I[a]) - 1))
                else:
                    sum_sigma = 0
                    for k1 in I[a]:
                        for k2 in I[b]:
                            sum_sigma += A[k1, a] * Sigma_condensed[k1, k2] * A[
                                k2, b]  # we note that we are only accessing A_I since k1, k2 are in I
                    C_condensed[a, b] = sum_sigma / (len(I[a]) * len(I[b]))
        W = A_I @ C_condensed @ A_I.T
        # print(A_I)

        '''dont know what to do here!!'''
        S_II = Sigma_hold_condensed[I_flat, :]
        S_II = S_II[:, I_flat]
        # print(off_diag_norm(W, S_II))
        if K == 1:
            losses[l] = 1
        else:
            losses[l] = estpure.off_diag_norm(W, S_II) / math.sqrt(
                max(len(I_flat) * (len(I_flat) - 1), 1))  # calculates cross-validation loss
        K_list[l] = K
        l += 1
        # print(losses[l-1])
    if return_type == "delta":
        return grid_deltas[np.argmin(losses)]

    else:
        print("CV loss minimized at", grid_deltas[np.argmin(losses)])
        ax1 = plt.subplot(211)
        ax1.plot(grid_deltas, losses)
        ax1.set_title("CV Loss vs Tuning Param Delta")

        ax2 = plt.subplot(212, sharex=ax1)
        ax2.plot(grid_deltas, K_list)
        ax2.set_title("Estimated K vs Tuning Param Delta")

        plt.show()
        # return K_list[np.argmin(losses)]
        return K_list, losses, grid_deltas

def make_AI(p, I, K, Sigma_hat):
    # this function constructs \hat{A}_{\hat I} using the partition of pure variables I and the estimated Sigma_hat. The set of indices in I may not be all at the top and may be shuffled.
    # Hence, we keep track of the positions of the pure variables (encoded in I) throughout the code.
    A = np.full((p,K), 0.0)
    lengthI = 0
    for Ia in I:
        lengthI += len(Ia)

    for a in range(K):
        Ia = I[a]
        list_Ia = list(Ia)
        A[list_Ia[0], a] = 1
        for j in range(1, len(Ia)):
            matrix_adding = Sigma_hat[list_Ia[0], list_Ia[0], :,: ] + Sigma_hat[list_Ia[0], list_Ia[j], :,: ]
            ev, _ = np.linalg.eigh(matrix_adding)
            adding_op_norm = max(ev)
            matrix_subtracting = Sigma_hat[list_Ia[0], list_Ia[0], :,: ] - Sigma_hat[list_Ia[0], list_Ia[j], :,: ]
            ev, _ = np.linalg.eigh(matrix_subtracting)
            subtracting_op_norm = max(ev)
            if adding_op_norm > subtracting_op_norm:
                A[list_Ia[j], a] = 1
            else:
                A[list_Ia[j], a] = -1
    list_I = []
    for Ia in I:
        list_I.extend(list(Ia))
    AI = A[list_I, :]
    return AI


def make_AJ(Sigma_hat, AI, I, K):
    # calculating number of pure variables
    lengthI = 0
    for pure_ind in I:
        lengthI += len(pure_ind)

    # constructing W
    W = AI.T @ AI

    # since the indices of AI might not be in order, we construct a temporary \hat A with \hat A_I and \hat A_J left as 0.
    m = Sigma_hat.shape[-1]
    p = Sigma_hat.shape[0]
    A = np.zeros((p, K), dtype=float)
    list_I = []
    for l in I:
        list_I += l
    for idx in range(lengthI):
        A[list_I[idx], :] = AI[idx, :]

        # constructing C
    C = np.zeros((K, K, m, m))
    for a in range(K):
        for b in range(K):
            if a == b:
                for i in I[a]:
                    for j in I[a]:
                        if i != j: C[a, a, :, :] += (A[i, a] * A[j, a]) * Sigma_hat[i, j, :, :]
                C[a, a, :, :] = C[a, a, :, :] / (len(I[a]) * (len(I[a]) - 1))
            else:
                for i in I[a]:
                    for j in I[b]:
                        C[a, b, :, :] += Sigma_hat[i, j, :, :] * (A[i, a] * A[j, b])
                C[a, b, :, :] = C[a, b, :, :] / (len(I[a]) * len(I[b]))

    # Solving regression for A_J:
    InverseMatrix = np.zeros((K, K))
    for s in range(K):
        for s2 in range(K):
            temp_entry = 0.0
            for k in range(K):
                temp_entry += np.trace(C[k, s2, :, :].T @ C[k, s, :, :])
            InverseMatrix[s, s2] = temp_entry

    # define U
    WinvAI = np.linalg.inv(W) @ AI.T  # this is a K times lengthI matrix
    U = np.zeros((K, p - lengthI, m, m))
    # extracting Sigma_IJ
    pure_indices = []
    for l in I:
        pure_indices += list(l)
    nonpure_indices = list(set(range(p)).difference(set(pure_indices)))

    for k in range(K):
        for j in range(p - lengthI):
            for i in range(lengthI):
                U[k, j, :, :] += WinvAI[k, i] * Sigma_hat[pure_indices[i], nonpure_indices[j], :, :]

    CU = np.zeros((K, p - lengthI))
    for s in range(K):
        for j in range(p - lengthI):
            temp_entry = 0.0
            for k in range(K):
                temp_entry += np.trace(U[k, j].T @ C[k, s])
            CU[s, j] = temp_entry

    AJ = np.linalg.inv(InverseMatrix) @ CU  # add a try-catch to this line in case the matrix is non-invertible.
    return AJ.T, C

# we now apply the functions defined above to get the estimated A and C
list_deltas = [] # finding optimal delta
for N in tqdm(range(25)):
    list_deltas.append(cross_val_mv_smooth(X))
optimDelta = statistics.median(list_deltas)

Sigma_hat = np.zeros((p, p, m, m), float) # calculating the estimated Sigma
for i in range(p):
    for j in range(p):
        Sigma_hat[i, j, :, :] = cov_fda(X[i, :, :], X[j, :, :])

est_I, est_K = estpure.pure_var(tensor_to_operator_norm(Sigma_hat), optimDelta)
est_AI = make_AI(p, est_I, est_K, Sigma_hat)
est_AJ, est_C = make_AJ(Sigma_hat, est_AI, est_I, est_K)

# NOTE: est_AJ is left as the raw, soft (unsparsified) estimate here.
# Thresholding into a hard/sparse loading matrix is now done downstream in
# evaluate.py, only for the specificity/sensitivity/rand-index metrics, and
# NOT for the L1/L2 estimation-error metrics. See evaluate.py:sparsify().

est_A = np.zeros((p, est_K), dtype=float)
list_I = []
for Ia in est_I:
    list_I += Ia
for pure_idx in range(len(list_I)):
    est_A[list_I[pure_idx], :] = est_AI[pure_idx, :]
non_pure_indices = list(set(range(p)).difference(set(list_I)))
for non_pure_idx in range(len(non_pure_indices)):
    est_A[non_pure_indices[non_pure_idx], :] = est_AJ[non_pure_idx, :]

output_name = "../results/factor_model_estA_%d.csv"%seed_number
pd.DataFrame(est_A).to_csv(output_name, index=False)
