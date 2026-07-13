# Simulation Study

Compares three clustering methods on 50 simulated functional datasets:

1. **Our method (funFMC)** — `python/factor_model_clus.py`
2. **Functional k-means** — `python/kmeans_func_clus.py`
3. **funHDDC** — `R/funHDDC_clus.R`

## Pipeline

```
simulate_data.py  -->  { factor_model_clus.py, kmeans_func_clus.py, funHDDC_clus.R }  -->  evaluate.py
   (writes to             (each reads data/, writes to results/)                          (reads results/,
    data/)                                                                                  writes results/)
```

Each clustering script and `simulate_data.py` take a single seed / array-task
ID as their command-line argument, and expect to be run from inside
`python/` (or `R/`), reading from `../data` and writing to `../results`.

## Running the full study (SLURM)

From `simulation-study/slurm/`, in order:

```bash
sbatch make_50_simulations.sbat   # simulates 50 datasets -> ../data/
sbatch run_fm.sbat                # our method            -> ../results/factor_model_estA_<id>.csv
sbatch run_kmeans.sbat            # functional k-means     -> ../results/kmeans_clus_<id>.txt
sbatch run_funHDDC.sbat           # funHDDC                -> ../results/funHDDC_result_<id>.txt
sbatch run_compare.sbat           # evaluate.py, once all of the above have finished
```

Fill in `#SBATCH --account=your_account_here` in each `.sbat` file with your
own cluster allocation before submitting. All four data/clustering jobs use
the array range `10-501:10` (i.e. seeds 10, 20, ..., 500 — 50 replicates),
so `run_compare.sbat` should only be submitted after all of those array jobs
have completed.

## Running a single replicate locally

```bash
make sim-one SEED=10       # from the repo root: simulate + all 3 methods for seed 10
make sim-evaluate           # aggregate metrics once results/ is populated
```

## Thresholding and evaluation

Our method performs **soft** (overlapping) clustering — `factor_model_clus.py`
writes the raw, unthresholded loading matrix to
`results/factor_model_estA_<id>.csv`. Thresholding into a hard/sparse matrix
(entries above 0.90 rounded to +-1, entries below 0.10 zeroed) happens only
in `evaluate.py`, and only for the specificity/sensitivity/rand-index
metrics — the L1/L2 estimation-error metrics are computed on the raw,
unthresholded estimate, since thresholding would distort the comparison
against the true (soft) loading matrix.

`evaluate.py`— computes
SP/SN/RI for all three methods plus L1/L2 error for the factor model in a
single pass, and writes:

- `results/comparison_results.csv` — metrics averaged across all 50 replicates
- `results/raw_sen_spec_ri.csv` — per-replicate metrics
