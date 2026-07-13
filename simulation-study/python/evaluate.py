# This file reads in the clustering output from all 3 methods (k-means,
# funHDDC, and our factor-model method) for the simulated datasets, and
# produces the average specificity (SP), sensitivity (SN), rand index (RI),
# and, for the factor model only, L1/L2 estimation error against the true
# loading matrix A.
#
# NOTE on thresholding: the factor model performs *soft* (overlapping)
# clustering, so est_A must be thresholded into a hard/sparse matrix before
# SP/SN/RI can be computed (they're defined on cluster *membership*, i.e.
# which entries are nonzero). The L1/L2 error, however, is computed on the
# raw, unthresholded est_A, since thresholding would distort the estimation
# error against the true (soft) loading matrix. This replaces the old
# eval_fm.py, whose functionality is now fully covered here.
import math
import os
from itertools import combinations, permutations

import numpy as np
import pandas as pd
from sklearn.metrics import rand_score

# ---------------------------------------------------------------------
# Config: how many simulated datasets, and which seed IDs, to evaluate.
# Matches the --array=10-501:10 convention used by the .sbat scripts.
# ---------------------------------------------------------------------
SIM_IDS = list(range(10, 501, 10))
DATA_DIR = "../data"
RESULTS_DIR = "../results"


def sparsify(A, upper=0.90, lower=0.05):
    """Threshold a soft loading matrix into a hard/sparse one.

    Entries with |value| > upper are rounded to +-1 and all other entries
    in that row are zeroed out (the variable is treated as "pure" to that
    cluster). Entries with |value| < lower are zeroed out entirely. This is
    the same thresholding logic that used to live at the end of
    factor_model_clus.py.
    """
    if len(A) == 0:
        return A
    A_sp = A.copy()
    for row in range(A_sp.shape[0]):
        for col in range(A_sp.shape[1]):
            if abs(A_sp[row, col]) > upper:
                A_sp[row, col] = math.copysign(1.0, A_sp[row, col])
                other_cols = [c for c in range(A_sp.shape[1]) if c != col]
                A_sp[row, other_cols] = 0.0
    for row in range(A_sp.shape[0]):
        for col in range(A_sp.shape[1]):
            if abs(A_sp[row, col]) < lower:
                A_sp[row, col] = 0.0
    return A_sp


def clustering_metrics(y_true, y_pred, K):
    """Sensitivity/specificity for hard clustering (k-means, funHDDC)."""
    if len(y_pred) == 0:
        return -1, -1

    tp, tn, fp, fn = 0, 0, 0, 0
    n_obs = len(y_pred)
    for i in range(n_obs):
        for j in range(i + 1, n_obs):
            if y_true[i] == y_true[j]:
                if y_pred[i] == y_pred[j]:
                    tp += 1
                else:
                    fn += 1
            else:
                if y_pred[i] == y_pred[j]:
                    fp += 1
                else:
                    tn += 1
    specificity = 0 if tn + fp == 0 else tn / (tn + fp)
    sensitivity = 0 if tp + fn == 0 else tp / (tp + fn)
    return specificity, sensitivity


def optimal_perm(est_A, true_A):
    """Find the signed column permutation of est_A minimizing Frobenius
    distance to true_A, so the two loading matrices can be compared
    entry-by-entry despite label-switching."""
    K = true_A.shape[1]
    base = np.ones((1, K), dtype=int)
    sign_perms = [base]
    for i in range(1, K + 1):
        for comb in combinations(range(K), i):
            perm = np.ones(K, dtype=int)
            perm[list(comb)] = -1
            sign_perms.append(perm[np.newaxis, :])
    all_sign_perms = np.vstack(sign_perms)

    all_col_perms = list(permutations(range(K)))
    prev_loss = np.linalg.norm(est_A - true_A, ord="fro")
    opt_perm = (None, None)

    for perm_indices in all_col_perms:
        permuted_A = est_A[:, perm_indices]
        for sign_vec in all_sign_perms:
            new_A = permuted_A * sign_vec
            curr_loss = np.linalg.norm(new_A - true_A, ord="fro")
            if curr_loss <= prev_loss:
                opt_perm = (perm_indices, sign_vec.copy())
                prev_loss = curr_loss
    return opt_perm


def clustering_metrics_overlapping(true_A, est_A, upper=0.90, lower=0.05):
    """SP/SN/RI (computed on the *thresholded* support of est_A) plus
    L1/L2 estimation error (computed on the *raw, unthresholded* est_A)."""
    if len(est_A) == 0:
        return -1, -1, -1, -1, -1
    if true_A.shape[0] != est_A.shape[0]:
        raise ValueError(
            "The true and estimated A matrices must have the same number "
            "of observations."
        )

    p, K = true_A.shape

    # --- SP / SN / RI on the thresholded support ---
    est_A_sparse = sparsify(est_A, upper, lower)
    TP, TN, FP, FN = 0.0, 0.0, 0.0, 0.0
    for i in range(p):
        for j in range(i + 1, p):
            true_i = set(np.where(true_A[i, :] != 0)[0])
            true_j = set(np.where(true_A[j, :] != 0)[0])
            est_i = set(np.where(est_A_sparse[i, :] != 0)[0])
            est_j = set(np.where(est_A_sparse[j, :] != 0)[0])
            if len(true_i.intersection(true_j)) > 0:
                if len(est_i.intersection(est_j)) > 0:
                    TP += 1
                else:
                    FN += 1
            else:
                if len(est_i.intersection(est_j)) > 0:
                    FP += 1
                else:
                    TN += 1
    specificity = 0 if TN + FP == 0 else TN / (TN + FP)
    sensitivity = 0 if TP + FN == 0 else TP / (TP + FN)
    rand_index = (TN + TP) / (TN + TP + FN + FP)

    # --- L1 / L2 error on the raw, unthresholded est_A ---
    if est_A.shape[1] == true_A.shape[1]:
        perm_index, sign_perm = optimal_perm(est_A, true_A)
        est_A_aligned = est_A[:, perm_index] * sign_perm
        l1_error = np.sum(np.abs(est_A_aligned - true_A)) / (p * K)
        l2_error = np.linalg.norm(est_A_aligned - true_A) / np.sqrt(p * K)
    else:
        l1_error, l2_error = -1, -1

    return specificity, sensitivity, rand_index, l1_error, l2_error


# this function aligns hard cluster labels with true labels (k-means/funHDDC)
def align_labels(y_true, y_pred, K):
    all_col_perms = list(permutations(range(K)))
    prev_loss = np.linalg.norm(y_true - y_pred)
    opt_perm = None
    for perm_indices in all_col_perms:
        permuted_y = np.array([perm_indices[label - 1] for label in y_pred]) + 1
        loss = np.linalg.norm(y_true - permuted_y)
        if loss < prev_loss:
            prev_loss = loss
            opt_perm = perm_indices
    if opt_perm is not None:
        return np.array([opt_perm[label - 1] for label in y_pred]) + 1
    return y_pred


def main():
    n_sims = len(SIM_IDS)
    kmeans_sp = np.zeros(n_sims)
    kmeans_sn = np.zeros(n_sims)
    kmeans_ri = np.zeros(n_sims)
    funhddc_sp = np.zeros(n_sims)
    funhddc_sn = np.zeros(n_sims)
    funhddc_ri = np.zeros(n_sims)
    fm_sp = np.zeros(n_sims)
    fm_sn = np.zeros(n_sims)
    fm_ri = np.zeros(n_sims)
    fm_l1 = np.zeros(n_sims)
    fm_l2 = np.zeros(n_sims)

    for idx, sim_id in enumerate(SIM_IDS):
        df = pd.read_csv(f"{DATA_DIR}/true_A_{sim_id}.csv")
        true_A = df.to_numpy()
        true_K = df.shape[1]
        true_labels = np.array(
            [np.where(df.iloc[i, :] == 1)[0][0] for i in range(df.shape[0])]
        ) + 1

        kmeans_path = f"{RESULTS_DIR}/kmeans_clus_{sim_id}.txt"
        with open(kmeans_path, "r") as f:
            kmeans_labels = np.array([int(x) for x in f.read().split("\n")[:-1]])

        funhddc_path = f"{RESULTS_DIR}/funHDDC_result_{sim_id}.txt"
        if os.path.exists(funhddc_path):
            with open(funhddc_path, "r") as f:
                funhddc_labels = np.array([int(x) for x in f.read().split("\n")[:-1]])
        else:
            funhddc_labels = np.array([])

        fm_path = f"{RESULTS_DIR}/factor_model_estA_{sim_id}.csv"
        est_A = pd.read_csv(fm_path).to_numpy() if os.path.exists(fm_path) else np.array([])

        kmeans_sp[idx], kmeans_sn[idx] = clustering_metrics(true_labels, kmeans_labels, true_K)
        kmeans_ri[idx] = rand_score(true_labels, kmeans_labels)

        funhddc_sp[idx], funhddc_sn[idx] = clustering_metrics(true_labels, funhddc_labels, true_K)
        funhddc_ri[idx] = -1 if funhddc_sp[idx] == -1 else rand_score(true_labels, funhddc_labels)

        fm_sp[idx], fm_sn[idx], fm_ri[idx], fm_l1[idx], fm_l2[idx] = clustering_metrics_overlapping(true_A, est_A)

    summary = {
        "Method": ["K-means", "funHDDC", "Factor Model"],
        "Specificity": [
            np.mean(kmeans_sp),
            np.mean(funhddc_sp[funhddc_sp != -1]),
            np.mean(fm_sp[fm_sp != -1]),
        ],
        "Sensitivity": [
            np.mean(kmeans_sn),
            np.mean(funhddc_sn[funhddc_sn != -1]),
            np.mean(fm_sn[fm_sn != -1]),
        ],
        "Rand Index": [
            np.mean(kmeans_ri),
            np.mean(funhddc_ri[funhddc_ri != -1]),
            np.mean(fm_ri[fm_ri != -1]),
        ],
        "L1 Error (Factor Model only)": [np.nan, np.nan, np.mean(fm_l1[fm_l1 != -1])],
        "L2 Error (Factor Model only)": [np.nan, np.nan, np.mean(fm_l2[fm_l2 != -1])],
    }
    pd.DataFrame(summary).to_csv(f"{RESULTS_DIR}/comparison_results.csv", index=False)

    raw = {
        "sim_id": SIM_IDS,
        "km_sp": kmeans_sp, "km_sn": kmeans_sn, "km_ri": kmeans_ri,
        "funHDDC_sp": funhddc_sp, "funHDDC_sn": funhddc_sn, "funHDDC_ri": funhddc_ri,
        "fm_sp": fm_sp, "fm_sn": fm_sn, "fm_ri": fm_ri,
        "fm_l1_error": fm_l1, "fm_l2_error": fm_l2,
    }
    pd.DataFrame(raw).to_csv(f"{RESULTS_DIR}/raw_sen_spec_ri.csv", index=False)
    print(f"Wrote {RESULTS_DIR}/comparison_results.csv and {RESULTS_DIR}/raw_sen_spec_ri.csv")


if __name__ == "__main__":
    main()
