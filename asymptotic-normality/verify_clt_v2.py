"""
Monte Carlo verification of the CLT covariance structure for the A_J estimator.

Model: X = A Z + E
  Z ~ N(0, C),  C_{ab} = c_scale * rho^{|a-b|}  (correlated factors)
  E ~ N(0, sigma^2 I)

Three checks:
  1. Empirical variance vs theoretical variance, entry by entry
  2. QQ plots for Gaussianity of sqrt(n)(A_J_hat - A_J)
  3. Empirical coverage of 95% confidence intervals

Main entry point: run_verification(config)
where config is a dict of hyperparameters (see DEFAULT_CONFIG below).
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import random

# ---------------------------------------------------------------
# 0. Default configuration
# ---------------------------------------------------------------
DEFAULT_CONFIG = dict(
    sim_type = 'simple',   # 'simple' or 'bing'

    # --- simple sim settings ---
    K        = 5,
    p        = 30,
    # n_pure   = 2,
    # sigma2   = 0.5,
    # c_scale  = 2.0,
    # rho      = 0.8,
    # overlap_weights = (0.6, 0.4),

    # --- shared settings ---
    B        = 2000,
    ns       = [100, 500, 1000, 2500],
    seed     = 42,
    qq_entries     = [(0,0), (5,2), (10,4), (15,1), (19,3)], # (j,s)th entry
    # of \widehat A_J. We show qqplots for \sqrt{n}(\widehat A_J - A_J) for
    # these entries.
    out_prefix     = "check",
)


# ---------------------------------------------------------------
# 1. Build true model from config
# ---------------------------------------------------------------
def build_model(config):
    if config.get('sim_type', 'simple') == 'bing':
        return _build_model_bing(config)
    else:
        return _build_model_simple(config)


def _build_model_simple(config):
    """Original simple simulation: structured pure/overlap variables."""
    K       = config['K']
    p       = config['p']
    n_pure  = 2
    sigma2  = 0.5

    assert n_pure * K < p, "Need p > n_pure * K for overlapping variables."

    A = np.zeros((p, K))
    for a in range(K):
        for r in range(n_pure):
            A[n_pure*a + r, a] = np.random.choice((1.0, -1.0))
    n_J = p - n_pure * K
    supp = [2, 3, 4, 5]
    for r in range(n_J):
        s_j = random.choice(supp)
        supp_j = random.sample(range(K), s_j)
        for k in supp_j:
            A[n_pure*K + r, k] = np.random.choice([-1, 1]) * 1.0 / s_j

    I_list = [list(range(n_pure*a, n_pure*a + n_pure)) for a in range(K)]
    I_flat = [i for Ia in I_list for i in Ia]
    J_flat = sorted(set(range(p)) - set(I_flat))
    n_J    = len(J_flat)
    A_I    = A[I_flat, :]
    A_J    = A[J_flat, :]

    C = np.array([[1.0 * (0.5**abs(a-b)) for b in range(K)]
                  for a in range(K)])
    Sigma = A @ C @ A.T + sigma2 * np.eye(p)

    W     = A_I.T @ A_I
    W_inv = np.linalg.inv(W)
    U     = W_inv @ A_I.T @ Sigma[np.ix_(I_flat, J_flat)]
    V     = C.T @ C
    V_inv = np.linalg.inv(V)

    A_full = np.zeros((p, K))
    A_full[I_flat, :] = A_I
    A_full[J_flat, :] = A_J

    return dict(
        A_true=A, A_I_true=A_I, A_J_true=A_J, A_full_true=A_full,
        C_true=C, Sigma_true=Sigma,
        W_true=W, W_inv=W_inv, U_true=U,
        V_true=V, V_inv=V_inv,
        I_list=I_list, I_flat=I_flat, J_flat=J_flat,
        K=K, p=p, n_J=n_J, sigma2=sigma2,
        sim_type='simple',
    )


def _build_model_bing(config):
    """
    Bing et al. (Section 5.2) simulation.
    Constructs A and C following simu1, but separates the data-generation
    step so that the true population quantities (A, C, Sigma) are fixed
    and only Sigma_hat varies across replications.
    """
    import random, math

    K = config['K']
    p = config['p']
    n_pure = 5

    # --- Factor covariance C ---
    C = np.full((K, K), 0.0)
    for i in range(K):
        C[i, i] = 2 + (i - 1) / (K - 1)
    for i in range(K):
        for j in range(K):
            if i != j:
                C[i, j] = ((-1)**(i+j)
                            * (0.3**abs(i-j))
                            * min(C[i, i], C[j, j]))

    # --- Loading matrix A (fixed by seed b) ---
    A = np.full((p, K), 0.0)
    for a in range(K):
        for r in range(n_pure):
            A[n_pure * a + r, a] = np.random.choice((1.0, -1.0))
    n_J = p - n_pure * K

    # A_J: overlapping variables
    supp = [2, 3, 4, 5]
    for j in range(n_J):
        s_j    = random.choice(supp)
        supp_j = random.sample(range(K), s_j)
        for k in supp_j:
            A[n_pure * K + j, k] = np.random.choice([-1, 1]) * 1.0 / s_j

    I_list = [list(range(n_pure * a, n_pure * a + n_pure)) for a in range(K)]
    I_flat = [i for Ia in I_list for i in Ia]
    J_flat = sorted(set(range(p)) - set(I_flat))
    A_I = A[I_flat, :]
    A_J = A[J_flat, :]

    # --- True noise variances (fixed once) ---
    np.random.seed(config['seed'])
    sigma_sqs = np.random.uniform(1.0, 3.0, size=p)

    # --- True Sigma ---
    Sigma = A @ C @ A.T + np.diag(sigma_sqs)

    # --- W, U, V ---
    W     = A_I.T @ A_I
    W_inv = np.linalg.inv(W)
    U     = W_inv @ A_I.T @ Sigma[np.ix_(I_flat, J_flat)]
    V     = C.T @ C
    V_inv = np.linalg.inv(V)

    A_full = np.zeros((p, K))
    A_full[I_flat, :] = A_I
    A_full[J_flat, :] = A_J

    return dict(
        A_true=A, A_I_true=A_I, A_J_true=A_J, A_full_true=A_full,
        C_true=C, Sigma_true=Sigma,
        W_true=W, W_inv=W_inv, U_true=U,
        V_true=V, V_inv=V_inv,
        I_list=I_list, I_flat=I_flat, J_flat=J_flat,
        K=K, p=p, n_J=n_J,
        sigma_sqs=sigma_sqs,   # heteroskedastic noise variances
        sim_type='bing',
    )

# ---------------------------------------------------------------
# 2. Generate n samples given the model
# ---------------------------------------------------------------
def generate_samples(n, model):
    K      = model['K']
    p      = model['p']
    A      = model['A_true']
    C      = model['C_true']

    Z  = np.random.multivariate_normal(np.zeros(K), C, size=n).T
    # heteroskedastic if sigma_sqs available, else homoskedastic
    if 'sigma_sqs' in model:
        E = np.array([np.random.normal(0, np.sqrt(sq), size=n)
                      for sq in model['sigma_sqs']])
    else:
        E = np.random.normal(0, np.sqrt(model['sigma2']), size=(p, n))
    X  = A @ Z + E
    Xc = X - X.mean(axis=1, keepdims=True)
    return (Xc @ Xc.T) / n

# ---------------------------------------------------------------
# 3. Estimate A_J given Sigma_hat and true A_I
# ---------------------------------------------------------------
def estimate_AJ(Sigma_hat, model):
    """
    Compute A_J_hat from the sample covariance Sigma_hat,
    using the true A_I (assumed known).

    Parameters
    ----------
    Sigma_hat : (p x p) sample covariance
    model     : dict returned by build_model

    Returns
    -------
    A_J_hat : (n_J x K) estimated overlapping loadings
    """
    K      = model['K']
    p      = model['p']
    A_I    = model['A_I_true']
    I_list = model['I_list']
    I_flat = model['I_flat']
    J_flat = model['J_flat']

    A_full = np.zeros((p, K))
    A_full[I_flat, :] = A_I

    # C_hat
    C_hat = np.zeros((K, K))
    for a in range(K):
        for b in range(K):
            Ia, Ib = I_list[a], I_list[b]
            if a == b:
                s = sum(A_full[i,a] * A_full[j,a] * Sigma_hat[i,j]
                        for i in Ia for j in Ia if i != j)
                C_hat[a,a] = s / (len(Ia) * (len(Ia) - 1))
            else:
                s = sum(A_full[i,a] * A_full[j,b] * Sigma_hat[i,j]
                        for i in Ia for j in Ib)
                C_hat[a,b] = s / (len(Ia) * len(Ib))

    # U_hat = W^{-1} A_I^T Sigma_{IJ}
    W_hat   = A_I.T @ A_I
    WinvAIT = np.linalg.inv(W_hat) @ A_I.T
    U_hat   = WinvAIT @ Sigma_hat[np.ix_(I_flat, J_flat)]  # K x n_J

    # V_hat = C_hat^T C_hat,  A_J_hat = V_hat^{-1} C_hat^T U_hat
    V_hat   = C_hat.T @ C_hat
    A_J_hat = (np.linalg.inv(V_hat) @ (C_hat.T @ U_hat)).T  # n_J x K

    return A_J_hat

# ---------------------------------------------------------------
# 4. Theoretical covariance from true parameters only
# ---------------------------------------------------------------
def compute_cov_theory(model):
    """
    Compute the theoretical asymptotic covariance of sqrt(n)(A_J_hat - A_J)
    using only true population quantities (no data).

    The linear map L(Delta)_{sj} = <w_final[s,jidx], Delta>_F encodes
    L = V^{-1}(L1 + L2 - G A_J^T) as a p x p real coefficient matrix.

    The covariance is then:
        Cov_theory[s1,j1,s2,j2]
            = tr(w_final[s1,j1] Sig w_final[s2,j2]^T Sig)
            + tr(w_final[s1,j1] Sig w_final[s2,j2]   Sig)
    which is the contraction of S_{ab,cd} = Sig_{ac}Sig_{bd} + Sig_{ad}Sig_{bc}
    (Gaussian fourth-moment formula) with w_final[s1,j1] and w_final[s2,j2].

    Parameters
    ----------
    model : dict returned by build_model

    Returns
    -------
    Cov_theory : array of shape (K, n_J, K, n_J)
                 Cov_theory[s1, j1, s2, j2] is the asymptotic covariance
                 between entries (s1,j1) and (s2,j2) of sqrt(n)(A_J_hat - A_J)
    """
    K        = model['K']
    n_J      = model['n_J']
    p        = model['p']
    Sigma    = model['Sigma_true']
    C        = model['C_true']
    U        = model['U_true']
    V_inv    = model['V_inv']
    A_full   = model['A_full_true']
    A_J      = model['A_J_true']
    I_list   = model['I_list']
    I_flat   = model['I_flat']
    J_flat   = model['J_flat']
    W_inv    = model['W_inv']
    A_I      = model['A_I_true']

    WinvAIT  = W_inv @ A_I.T   # K x |I|

    # --- Step 1: build w_raw[s, jidx] (p x p) ---
    # w_raw encodes (L1 + L2 - G A_J^T)(Delta)_{sj} = <w_raw[s,jidx], Delta>_F
    w_raw = np.zeros((K, n_J, p, p))

    for s in range(K):
        for jidx in range(n_J):
            j = J_flat[jidx]
            w = np.zeros((p, p))

            # L1: coefficient of Delta[i0, j] is sum_k C[k,s] * WinvAIT[k, ii]
            for k in range(K):
                for ii, i_idx in enumerate(I_flat):
                    w[i_idx, j] += C[k, s] * WinvAIT[k, ii]

            # L2: coefficient of Delta[i1, i2] is U[k,jidx] * A[i1,k]*A[i2,s]/norm
            for k in range(K):
                u_val = U[k, jidx]
                Ik, Is = I_list[k], I_list[s]
                if k == s:
                    norm = len(Ik) * (len(Ik) - 1)
                    for i1 in Ik:
                        for i2 in Ik:
                            if i1 != i2:
                                w[i1, i2] += (u_val * A_full[i1,k]
                                              * A_full[i2,s] / norm)
                else:
                    norm = len(Ik) * len(Is)
                    for i1 in Ik:
                        for i2 in Is:
                            w[i1, i2] += (u_val * A_full[i1,k]
                                          * A_full[i2,s] / norm)

            # G: subtract [G(Delta) A_J^T]_{s,jidx}
            # = sum_t A_J[jidx,t] sum_k (T(Delta)_{ks}*C[k,t] + C[k,s]*T(Delta)_{kt})
            for t in range(K):
                ajt  = A_J[jidx, t]
                for k in range(K):
                    Ik, Is, It = I_list[k], I_list[s], I_list[t]
                    c_kt = C[k, t]
                    c_ks = C[k, s]
                    # T(Delta)_{ks} * C[k,t]
                    if k == s:
                        norm = len(Ik) * (len(Ik) - 1)
                        for i1 in Ik:
                            for i2 in Ik:
                                if i1 != i2:
                                    w[i1,i2] -= (ajt * c_kt
                                                 * A_full[i1,k]
                                                 * A_full[i2,s] / norm)
                    else:
                        norm = len(Ik) * len(Is)
                        for i1 in Ik:
                            for i2 in Is:
                                w[i1,i2] -= (ajt * c_kt
                                             * A_full[i1,k]
                                             * A_full[i2,s] / norm)
                    # C[k,s] * T(Delta)_{kt}
                    if k == t:
                        norm = len(Ik) * (len(Ik) - 1)
                        for i1 in Ik:
                            for i2 in Ik:
                                if i1 != i2:
                                    w[i1,i2] -= (ajt * c_ks
                                                 * A_full[i1,k]
                                                 * A_full[i2,t] / norm)
                    else:
                        norm = len(Ik) * len(It)
                        for i1 in Ik:
                            for i2 in It:
                                w[i1,i2] -= (ajt * c_ks
                                             * A_full[i1,k]
                                             * A_full[i2,t] / norm)
            w_raw[s, jidx] = w

    # --- Step 2: apply V^{-1} sandwich ---
    # w_final[s,jidx] = sum_{s'} V_inv[s,s'] w_raw[s',jidx]
    w_final = np.tensordot(V_inv, w_raw, axes=([1], [0]))  # K x n_J x p x p

    # --- Step 3: covariance via Gaussian S formula ---
    # Cov[s1,j1,s2,j2] = tr(w1 Sig w2^T Sig) + tr(w1 Sig w2 Sig)
    Cov_theory = np.zeros((K, n_J, K, n_J))
    for s1 in range(K):
        for j1 in range(n_J):
            w1  = w_final[s1, j1]
            w1S = w1 @ Sigma              # precompute p x p
            for s2 in range(K):
                for j2 in range(n_J):
                    w2 = w_final[s2, j2]
                    Cov_theory[s1,j1,s2,j2] = (
                        np.trace(w1S @ w2.T @ Sigma)
                        + np.trace(w1S @ w2   @ Sigma)
                    )

    return Cov_theory

# ---------------------------------------------------------------
# 5. Monte Carlo runner
# ---------------------------------------------------------------
def run_mc(n, B, model):
    """
    Run B Monte Carlo replications at sample size n.

    Returns
    -------
    diffs : (B, n_J, K) array of sqrt(n)*(A_J_hat - A_J_true)
    """
    n_J     = model['n_J']
    K       = model['K']
    A_J_true = model['A_J_true']
    diffs   = np.zeros((B, n_J, K))
    for b in range(B):
        Sigma_hat   = generate_samples(n, model)
        A_J_hat     = estimate_AJ(Sigma_hat, model)
        diffs[b]    = np.sqrt(n) * (A_J_hat - A_J_true)
    return diffs

# ---------------------------------------------------------------
# 6. Plotting functions
# ---------------------------------------------------------------

def plot_variance_and_covariance(results, Cov_theory, ns, model, out_prefix):
    K, n_J = model['K'], model['n_J']

    # --- Theoretical values ---
    # Diagonal (variances): shape (n_J, K)
    th_diag = np.array([[Cov_theory[s, jidx, s, jidx]
                         for s in range(K)]
                        for jidx in range(n_J)])

    # Off-diagonal (cross-covariances): all (s1,j1) != (s2,j2) pairs
    # Flatten to 1D arrays for scatter plot
    th_cross, th_diag_flat = [], []
    idx_diag, idx_cross = [], []
    for jidx1 in range(n_J):
        for s1 in range(K):
            for jidx2 in range(n_J):
                for s2 in range(K):
                    val = Cov_theory[s1, jidx1, s2, jidx2]
                    if jidx1 == jidx2 and s1 == s2:
                        th_diag_flat.append(val)
                    else:
                        th_cross.append(val)
    th_diag_flat = np.array(th_diag_flat)
    th_cross     = np.array(th_cross)

    fig, axes = plt.subplots(1, len(ns), figsize=(5*len(ns), 5))
    if len(ns) == 1:
        axes = [axes]

    for ax, n in zip(axes, ns):
        diffs     = results[n]                          # (B, n_J, K)
        B         = diffs.shape[0]
        diffs_flat = diffs.reshape(B, n_J * K)          # (B, n_J*K)
        emp_cov   = np.cov(diffs_flat, rowvar=False)    # (n_J*K, n_J*K)

        # Extract diagonal and off-diagonal empirical values
        # using same ordering: index = jidx*K + s
        emp_diag_flat, emp_cross = [], []
        for jidx1 in range(n_J):
            for s1 in range(K):
                for jidx2 in range(n_J):
                    for s2 in range(K):
                        val = emp_cov[jidx1*K + s1, jidx2*K + s2]
                        if jidx1 == jidx2 and s1 == s2:
                            emp_diag_flat.append(val)
                        else:
                            emp_cross.append(val)
        emp_diag_flat = np.array(emp_diag_flat)
        emp_cross     = np.array(emp_cross)

        # Plot diagonal (variances) in blue, cross-cov in orange
        ax.scatter(th_diag_flat, emp_diag_flat,
                   alpha=0.7, s=30, color='steelblue',
                   label='Variance (diagonal)')
        ax.scatter(th_cross, emp_cross,
                   alpha=0.3, s=10, color='darkorange',
                   label='Cross-covariance')

        # y=x reference line
        all_vals = np.concatenate([th_diag_flat, th_cross,
                                   emp_diag_flat, emp_cross])
        lim = [all_vals.min() - 0.05*abs(all_vals.min()),
               all_vals.max() + 0.05*abs(all_vals.max())]
        ax.plot(lim, lim, 'r--', lw=1.5)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Theoretical covariance", size=10)
        ax.set_ylabel("Empirical covariance", size=10)
        ax.set_title(f"n = {n}", size=14)
        #ax.legend(fontsize=8)

        #corr_diag  = np.corrcoef(th_diag_flat, emp_diag_flat)[0,1]
        #corr_cross = np.corrcoef(th_cross, emp_cross)[0,1]
        #ax.text(0.05, 0.92,
        #        f"r (var) = {corr_diag:.3f}\nr (cross) = {corr_cross:.3f}",
        #        transform=ax.transAxes, fontsize=9)

    #fig.suptitle("Check 1: Empirical vs Theoretical Covariance\n"
    #             "(blue = variances, orange = cross-covariances)",
    #             fontsize=12)
    plt.tight_layout()
    path = f"{out_prefix}_covariance_{model['sim_type']}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_qq(results, Cov_theory, ns, model, config):
    entries    = config['qq_entries']
    out_prefix = config['out_prefix']
    fig, axes  = plt.subplots(len(entries), len(ns),
                               figsize=(5*len(ns), 4*len(entries)))
    if len(entries) == 1:
        axes = axes[np.newaxis, :]
    for row, (jidx, s) in enumerate(entries):
        th_std = np.sqrt(Cov_theory[s, jidx, s, jidx])
        for col, n in enumerate(ns):
            ax  = axes[row, col]
            obs = results[n][:, jidx, s] / th_std
            stats.probplot(obs, dist="norm", plot=ax)
            ax.set_title(f"$(j={jidx}, s={s})$, $n={n}$", fontsize=9)
            ax.get_lines()[0].set(markersize=2, alpha=0.4)
            ax.get_lines()[1].set(color='red', lw=1.5)
    fig.suptitle("Check 2: QQ plots of standardized "
                 "$\\sqrt{n}(\\hat A_{J,sj} - A_{J,sj})$", fontsize=12)
    plt.tight_layout()
    path = f"{out_prefix}_qqplots_{model['sim_type']}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_coverage(results, Cov_theory, ns, model, out_prefix):
    K, n_J   = model['K'], model['n_J']
    coverage = np.zeros((len(ns), n_J, K))
    for ni, n in enumerate(ns):
        for jidx in range(n_J):
            for s in range(K):
                th_std  = np.sqrt(Cov_theory[s, jidx, s, jidx])
                covered = np.abs(results[n][:, jidx, s]) <= 1.96 * th_std
                coverage[ni, jidx, s] = covered.mean()
    fig, axes = plt.subplots(1, len(ns), figsize=(5*len(ns), 4), sharey=True)
    if len(ns) == 1:
        axes = [axes]
    for ax, n, ni in zip(axes, ns, range(len(ns))):
        cov_flat = coverage[ni].ravel()
        ax.hist(cov_flat, bins=15, range=(0.80, 1.0),
                edgecolor='black', alpha=0.7)
        ax.axvline(0.95, color='red', linestyle='--', lw=2, label='Nominal 95%')
        ax.set_xlabel("Empirical coverage", size = 10)
        ax.set_title(f"n = {n}", size = 14)
                     #f"\nMean = {cov_flat.mean():.3f}, "
                     #f"Min = {cov_flat.min():.3f}")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Number of entries", size = 10)
    #fig.suptitle("Check 3: Empirical coverage of nominal 95% CIs",
    # fontsize=12)
    plt.tight_layout()
    path = f"{out_prefix}_coverage_{model['sim_type']}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")
    return coverage

# ---------------------------------------------------------------
# 7. Summary table
# ---------------------------------------------------------------
def print_summary(results, Cov_theory, coverage, ns, model):
    th_mean = np.array([[Cov_theory[s, jidx, s, jidx]
                         for s in range(model['K'])]
                        for jidx in range(model['n_J'])]).mean()
    print(f"\n{'n':>6}  {'Mean emp var':>14}  {'Mean th var':>12}  "
          f"{'Ratio':>7}  {'Mean coverage':>14}")
    for ni, n in enumerate(ns):
        emp_var  = results[n].var(axis=0).mean()
        cov_mean = coverage[ni].mean()
        print(f"{n:>6}  {emp_var:>14.4f}  {th_mean:>12.4f}  "
              f"{emp_var/th_mean:>7.3f}  {cov_mean:>14.3f}")

# ---------------------------------------------------------------
# 8. Main entry point
# ---------------------------------------------------------------
def run_verification(config=None):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    np.random.seed(cfg['seed'])

    print("=== Building model ===")
    model = build_model(cfg)
    sim   = model['sim_type']
    if sim == 'simple':
        print(f"  [simple] K={model['K']}, p={model['p']}, n_J="
              f"{model['n_J']}, A = {model['A_true']}")
    else:
        print(f"  [bing]   K={model['K']}, p={model['p']}, n_J={model['n_J']}")

    print("\n=== Computing theoretical covariance ===")
    Cov_theory = compute_cov_theory(model)
    print("  Done.")

    print("\n=== Running Monte Carlo ===")
    results = {}
    for n in cfg['ns']:
        print(f"  n={n}, B={cfg['B']}...")
        results[n] = run_mc(n, cfg['B'], model)
    print("  Done.")

    print("\n=== Plotting ===")
    prefix   = cfg['out_prefix']
    plot_variance_and_covariance(results, Cov_theory, cfg['ns'], model, prefix)
    plot_qq(results, Cov_theory, cfg['ns'], model, cfg)
    coverage = plot_coverage(results, Cov_theory, cfg['ns'], model, prefix)
    print_summary(results, Cov_theory, coverage, cfg['ns'], model)

    return model, Cov_theory, results, coverage


# ---------------------------------------------------------------
# 9. Run with defaults when executed as a script
# ---------------------------------------------------------------
if __name__ == "__main__":
    run_verification(dict(sim_type="simple", p=60))