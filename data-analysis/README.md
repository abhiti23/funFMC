# Data Analysis: AOMIC Resting-State fMRI

Applies our method to Schaefer-200-parcellated resting-state fMRI time series
from the AOMIC dataset.

The code is written assuming
`data/aomic_data_200.npy` (a `(p=200, n_subjects, m_timepoints)` array of
parcel-averaged time series) already exists locally — data is available on request.

## Pipeline

```
one_cv_gl.py (x10 seeds, SLURM)  -->  cat delta_*.txt > all_deltas.txt  -->  final_clus.py  -->  plot_*.py
   picks a CV delta per seed          (manual aggregation step)              estimates A, K       visualizes A
   -> results/delta_<seed>.txt                                               -> results/est_A.csv  on brain atlases
                                                                              -> results/est_K.txt
                                                                              -> results/est_I.csv
```

### 1. Cross-validation (on the cluster)

```bash
cd slurm
sbatch run_aomic_cvs.sbat
```

Fill in `#SBATCH --account=your_account_here` first. This runs
`one_cv_gl.py` for 10 seeds (array `5-50:5`) and writes
`results/delta_5.txt`, `results/delta_10.txt`, ..., `results/delta_50.txt`.

### 2. Aggregate the deltas

Once all 10 array jobs finish:

```bash
cd results
cat delta_*.txt > all_deltas.txt
```

(or `make aomic-aggregate` from the repo root). `final_clus.py` takes the
median of whatever's in `all_deltas.txt` as the tuning parameter, so this
file must exist before step 3.

### 3. Final clustering (can run locally)

```bash
cd python
python3 final_clus.py
```

This reads `../data/aomic_data_200.npy` and `../results/all_deltas.txt`.
It computes (and caches, to `../results/Sigma_condensed_aomic.csv`) the
condensed covariance matrix used for pure-variable selection, estimates the
loading matrix `A` and cluster count `K`, thresholds the impure-variable
block (`sparsify_AJ`, same 0.90/0.10 convention as the simulation study),
and writes `results/est_A.csv`, `results/est_K.txt`, `results/est_I.csv`.

If `Sigma_condensed_aomic.csv` already exists in `results/`, it's loaded
from cache rather than recomputed (this step is the slow part of the
pipeline). Delete it if you want it recomputed from scratch.

### 4. Visualization (local)

```bash
python3 plot_yeo17_network_loadings.py   # bar/lollipop plots by Yeo-17 network
python3 plot_surface_clusters.py         # cortical surface maps per cluster
```

Both read `results/est_A.csv` and apply their own display threshold
(`THRESHOLD = 0.10`) independently of the 0.90/0.10 thresholding in
`final_clus.py` — that's for display only and doesn't change the saved
`est_A.csv`.

## Note on paths

All the code now reads from and
writes to the same `data-analysis/results/` folder regardless of where it's
run, so the pipeline works end-to-end on a single machine as well as split
across cluster + local steps.
