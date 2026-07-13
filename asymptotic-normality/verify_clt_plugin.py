import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import random

DEFAULT_CONFIG = dict(
    sim_type = 'simple',
    K        = 5,
    p        = 30,
    B        = 2000,
    ns       = [100, 500, 1000, 2500],
    seed     = 42,
    qq_entries     = [(0,0), (5,2), (10,4), (15,1), (19,3)],
    out_prefix     = "check",
)

def build_model(config):
    if config.get('sim_type', 'simple') == 'bing':
        return _build_model_bing(config)
    else:
        return _build_model_simple(config)

def _build_model_simple(config):
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
    K = config['K']
    p = config['p']
    n_pure = 5

    C = np.full((K, K), 0.0)
    for i in range(K):
        C[i, i] = 2 + (i - 1) / (K - 1)
    for i in range(K):
        for j in range(K):
            if i != j:
                C[i, j] = ((-1)**(i+j)
                            * (0.3**abs(i-j))
                            * min(C[i, i], C[j, j]))

    A = np.full((p, K), 0.0)
    for a in range(K):
        for r in range(n_pure):
            A[n_pure * a + r, a] = np.random.choice((1.0, -1.0))
    n_J = p - n_pure * K

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

    np.random.seed(config['seed'])
    sigma_sqs = np.random.uniform(1.0, 3.0, size=p)
    Sigma = A @ C @ A.T + np.diag(sigma_sqs)

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
        sigma_sqs=sigma_sqs,
        sim_type='bing',
    )

def generate_samples(n, model):
    K      = model['K']
    p      = model['p']
    A      = model['A_true']
    C      = model['C_true']

    Z  = np.random.multivariate_normal(np.zeros(K), C, size=n).T
    if 'sigma_sqs' in model:
        E = np.array([np.random.normal(0, np.sqrt(sq), size=n)
                      for sq in model['sigma_sqs']])
    else:
        E = np.random.normal(0, np.sqrt(model['sigma2']), size=(p, n))
    X  = A @ Z + E
    Xc = X - X.mean(axis=1, keepdims=True)
    return (Xc @ Xc.T) / n, Xc

def estimate_AJ(Sigma_hat, model):
    K      = model['K']
    p      = model['p']
    A_I    = model['A_I_true']
    I_list = model['I_list']
    I_flat = model['I_flat']
    J_flat = model['J_flat']

    A_full = np.zeros((p, K))
    A_full[I_flat, :] = A_I

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

    W_hat   = A_I.T @ A_I
    WinvAIT = np.linalg.inv(W_hat) @ A_I.T
    U_hat   = WinvAIT @ Sigma_hat[np.ix_(I_flat, J_flat)]

    V_hat   = C_hat.T @ C_hat
    A_J_hat = (np.linalg.inv(V_hat) @ (C_hat.T @ U_hat)).T

    return A_J_hat, C_hat, U_hat, V_hat

def compute_plugin_se(Xc, Sigma_hat, A_J_hat, C_hat, U_hat, V_hat, model):
    K      = model['K']
    p      = model['p']
    n_J    = model['n_J']
    I_list = model['I_list']
    I_flat = model['I_flat']
    J_flat = model['J_flat']
    A_I    = model['A_I_true']
    n      = Xc.shape[1]

    A_full_hat = np.zeros((p, K))
    A_full_hat[I_flat, :] = A_I
    A_full_hat[J_flat, :] = A_J_hat

    W_hat    = A_I.T @ A_I
    WinvAIT  = np.linalg.inv(W_hat) @ A_I.T
    V_hat_inv = np.linalg.inv(V_hat)

    w_raw = np.zeros((K, n_J, p, p))

    for s in range(K):
        for jidx in range(n_J):
            j = J_flat[jidx]
            w = np.zeros((p, p))

            for k in range(K):
                for ii, i_idx in enumerate(I_flat):
                    w[i_idx, j] += C_hat[k, s] * WinvAIT[k, ii]

            for k in range(K):
                u_val = U_hat[k, jidx]
                Ik, Is = I_list[k], I_list[s]
                if k == s:
                    norm = len(Ik) * (len(Ik) - 1)
                    for i1 in Ik:
                        for i2 in Ik:
                            if i1 != i2:
                                w[i1, i2] += (u_val * A_full_hat[i1, k]
                                              * A_full_hat[i2, s] / norm)
                else:
                    norm = len(Ik) * len(Is)
                    for i1 in Ik:
                        for i2 in Is:
                            w[i1, i2] += (u_val * A_full_hat[i1, k]
                                          * A_full_hat[i2, s] / norm)

            for t in range(K):
                ajt  = A_J_hat[jidx, t]
                for k in range(K):
                    Ik, Is, It = I_list[k], I_list[s], I_list[t]
                    c_kt = C_hat[k, t]
                    c_ks = C_hat[k, s]
                    if k == s:
                        norm = len(Ik) * (len(Ik) - 1)
                        for i1 in Ik:
                            for i2 in Ik:
                                if i1 != i2:
                                    w[i1, i2] -= (ajt * c_kt
                                                  * A_full_hat[i1, k]
                                                  * A_full_hat[i2, s] / norm)
                    else:
                        norm = len(Ik) * len(Is)
                        for i1 in Ik:
                            for i2 in Is:
                                w[i1, i2] -= (ajt * c_kt
                                              * A_full_hat[i1, k]
                                              * A_full_hat[i2, s] / norm)
                    if k == t:
                        norm = len(Ik) * (len(Ik) - 1)
                        for i1 in Ik:
                            for i2 in Ik:
                                if i1 != i2:
                                    w[i1, i2] -= (ajt * c_ks
                                                  * A_full_hat[i1, k]
                                                  * A_full_hat[i2, t] / norm)
                    else:
                        norm = len(Ik) * len(It)
                        for i1 in Ik:
                            for i2 in It:
                                w[i1, i2] -= (ajt * c_ks
                                              * A_full_hat[i1, k]
                                              * A_full_hat[i2, t] / norm)
            w_raw[s, jidx] = w

    w_final = np.tensordot(V_hat_inv, w_raw, axes=([1], [0]))

    outer_all = (Xc[:, np.newaxis, :] * Xc[np.newaxis, :, :])
    outer_centered = outer_all - Sigma_hat[:, :, np.newaxis]

    xi = np.einsum('sjab, abn -> sjn', w_final, outer_centered)

    var_hat = (xi**2).mean(axis=2)
    se      = np.sqrt(var_hat)

    return se.T

def compute_cov_theory(model):
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

    WinvAIT  = W_inv @ A_I.T

    w_raw = np.zeros((K, n_J, p, p))

    for s in range(K):
        for jidx in range(n_J):
            j = J_flat[jidx]
            w = np.zeros((p, p))

            for k in range(K):
                for ii, i_idx in enumerate(I_flat):
                    w[i_idx, j] += C[k, s] * WinvAIT[k, ii]

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

            for t in range(K):
                ajt  = A_J[jidx, t]
                for k in range(K):
                    Ik, Is, It = I_list[k], I_list[s], I_list[t]
                    c_kt = C[k, t]
                    c_ks = C[k, s]
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

    w_final = np.tensordot(V_inv, w_raw, axes=([1], [0]))

    Cov_theory = np.zeros((K, n_J, K, n_J))
    for s1 in range(K):
        for j1 in range(n_J):
            w1  = w_final[s1, j1]
            w1S = w1 @ Sigma
            for s2 in range(K):
                for j2 in range(n_J):
                    w2 = w_final[s2, j2]
                    Cov_theory[s1,j1,s2,j2] = (
                        np.trace(w1S @ w2.T @ Sigma)
                        + np.trace(w1S @ w2   @ Sigma)
                    )

    return Cov_theory

def run_mc(n, B, model):
    n_J       = model['n_J']
    K         = model['K']
    A_J_true  = model['A_J_true']
    diffs     = np.zeros((B, n_J, K))
    se_plugin = np.zeros((B, n_J, K))
    Cov_theory = compute_cov_theory(model)

    for b in range(B):
        Sigma_hat, Xc = generate_samples(n, model)
        A_J_hat, C_hat, U_hat, V_hat = estimate_AJ(Sigma_hat, model)
        diff_b = np.sqrt(n) * (A_J_hat - A_J_true)
        se_b   = compute_plugin_se(Xc, Sigma_hat, A_J_hat,
                                    C_hat, U_hat, V_hat, model)
        diffs[b]     = diff_b
        se_plugin[b] = se_b

        # Add this inside run_mc after the b==0 block:
        if b == 0:
            ratios = se_b / np.sqrt(np.array([[Cov_theory[s, jidx, s, jidx]
                                               for s in range(K)]
                                              for jidx in range(n_J)]))
            print(f"  Mean ratio across all entries: {ratios.mean():.3f}, "
                  f"std: {ratios.std():.3f}, "
                  f"max: {ratios.max():.3f}, "
                  f"min: {ratios.min():.3f}")

    return diffs, se_plugin

def plot_plugin_coverage(diffs, se_plugins, ns, model, out_prefix):
    K, n_J = model['K'], model['n_J']
    cov_plugin = np.zeros((len(ns), n_J, K))
    for ni, n in enumerate(ns):
        for jidx in range(n_J):
            for s in range(K):
                covered = (np.abs(diffs[n][:, jidx, s])
                           <= 1.96 * se_plugins[n][:, jidx, s])
                cov_plugin[ni, jidx, s] = covered.mean()

    fig, axes = plt.subplots(1, len(ns), figsize=(5*len(ns), 4), sharey=True)
    if len(ns) == 1:
        axes = [axes]

    for ax, n, ni in zip(axes, ns, range(len(ns))):
        cov_flat = cov_plugin[ni].ravel()
        ax.hist(cov_flat, bins=15, range=(0.75, 1.0),
                edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(0.95, color='red', linestyle='--', lw=2, label='Nominal 95%')
        ax.set_xlabel("Empirical coverage", size = 10)
        ax.set_title(f"n = {n}", size = 14)
                     #f"\nMean = {cov_flat.mean():.3f}, Min =
        # {cov_flat.min():.3f}")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Number of entries", size = 10)
    # fig.suptitle("Plug-in CI coverage", fontsize=12)
    plt.tight_layout()
    path = f"{out_prefix}_plugin_coverage_{model['sim_type']}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")
    return cov_plugin

def plot_plugin_qq(results, se_plugins, ns, model, config):
    entries    = config['qq_entries']
    out_prefix = config['out_prefix']
    fig, axes  = plt.subplots(len(entries), len(ns),
                               figsize=(5*len(ns), 4*len(entries)))
    if len(entries) == 1:
        axes = axes[np.newaxis, :]
    for row, (jidx, s) in enumerate(entries):
        for col, n in enumerate(ns):
            ax  = axes[row, col]
            num = results[n][:, jidx, s]
            den = se_plugins[n][:, jidx, s]
            obs = num / den
            stats.probplot(obs, dist="norm", plot=ax)
            ax.set_title(f"$(j={jidx}, s={s})$, $n={n}$", fontsize=14)
            ax.get_lines()[0].set(markersize=2, alpha=0.4)
            ax.get_lines()[1].set(color='red', lw=1.5)
    #fig.suptitle("Plug-in QQ plots", fontsize=11)
    plt.tight_layout()
    path = f"{out_prefix}_plugin_qqplots_{model['sim_type']}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")

def plot_coverage(results, Cov_theory, ns, model):
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
        ax.set_title(f"n = {n}", size=14)
                     #f"\nMean = {cov_flat.mean():.3f}, "
                     #f"Min = {cov_flat.min():.3f}")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Number of entries", size = 10)
    fig.suptitle("Check 3: Empirical coverage of nominal 95% CIs", fontsize=12)
    plt.tight_layout()
    path = f"VERIFY_coverage_{model['sim_type']}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")
    return coverage

def print_summary_theoretical(results, Cov_theory, coverage, ns, model):
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

def run_verification(config=None):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    np.random.seed(cfg['seed'])

    print("=== Building model ===")
    model = build_model(cfg)
    print(f"  [{model['sim_type']}] K={model['K']}, p={model['p']}, n_J={model['n_J']}")

    print("\n=== Running Monte Carlo ===")
    results    = {}
    se_plugins = {}
    for n in cfg['ns']:
        print(f"  n={n}, B={cfg['B']}...")
        diffs, se_plug = run_mc(n, cfg['B'], model)
        results[n]    = diffs
        se_plugins[n] = se_plug
    print("  Done.")

    print("\n=== Computing theoretical covariance ===")
    Cov_theory = compute_cov_theory(model)
    print("  Done.")
    coverage = plot_coverage(results, Cov_theory, cfg['ns'], model)
    print_summary_theoretical(results, Cov_theory, coverage, cfg['ns'], model)


    print("\n=== Plotting ===")
    prefix     = cfg['out_prefix']
    cov_plugin = plot_plugin_coverage(results, se_plugins, cfg['ns'], model, prefix)
    plot_plugin_qq(results, se_plugins, cfg['ns'], model, cfg)

    print("\n=== Plugin coverage summary ===")
    for ni, n in enumerate(cfg['ns']):
        print(f"  n={n}: mean={cov_plugin[ni].mean():.3f}, min={cov_plugin[ni].min():.3f}")

    return model, results, se_plugins, cov_plugin

if __name__ == "__main__":
    run_verification(dict(sim_type="bing", p=100, ns=[100, 500, 1000, 2500],
                          B=2000))