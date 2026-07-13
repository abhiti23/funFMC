# Asymptotic Normality Verification

Monte Carlo checks that the CLT for the non-pure part of the loading matrix
`A_J` behaves as the theory predicts. Both scripts share the same simulation
model (`build_model`, `generate_samples`, `estimate_AJ`,
`compute_cov_theory`) but check different things, so they're kept as two
separate, independently runnable scripts rather than merged:

- **`verify_clt_v2.py`** — checks the CLT itself: is
  `sqrt(n) * (A_hat_J - A_J)` asymptotically normal with the *theoretical*
  (oracle) covariance? Produces variance-comparison plots, QQ plots against
  the theoretical covariance, and coverage of theoretical 95% CIs.

- **`verify_clt_plugin.py`** — checks whether a practical, *data-driven*
  plug-in standard error estimator (the one you'd actually compute without
  knowing the truth) gives correct coverage, by comparing it against the
  theoretical covariance from the same model.

## Running

```bash
python3 verify_clt_v2.py
python3 verify_clt_plugin.py
```

Or via the Makefile from the repo root: `make clt-v2`, `make clt-plugin`.

Each script's `if __name__ == "__main__"` block calls `run_verification`
with a config dict (`sim_type`, `K`, `p`, `B`, `ns`, `seed`, etc. — see
`DEFAULT_CONFIG` at the top of each file for all options). Output figures
are saved to the working directory the script is run from; the `results/`
folder in this directory is provided as a convenient place to run them from
if you'd rather keep figures out of the repo root.
