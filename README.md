# Overlapping Functional Clustering via Structured Factor Models

Code accompanying the paper titled "funFMC: Overlapping Clustering for Functional Data". This repository
contains three self-contained components:

```
.
├── common/                   shared library code (the LOVE-style pure-variable
│                              selection algorithm), imported by both the
│                              simulation study and the data analysis
├── simulation-study/          compares our method against functional k-means
│                              and funHDDC across 50 simulated replicates
├── asymptotic-normality/      Monte Carlo verification that the CLT for our
│                              estimator, and a practical plug-in standard
│                              error, behave as the theory predicts
├── data-analysis/             applies our method to the AOMIC resting-state
│                              fMRI dataset
├── requirements.txt
├── Makefile
└── .gitignore
```

Each component has its own `README.md` with details specific to that piece.

## Data

The simulation study generates
its own synthetic data, so nothing needs to be supplied there. The data
analysis component is written assuming `data-analysis/data/aomic_data_200.npy`
(Schaefer-200-parcellated AOMIC resting-state time series) is available
locally; see `data-analysis/README.md`.

## Setup

```bash
pip install -r requirements.txt
```

The simulation study's funHDDC step additionally requires R with the
[`funHDDC`](https://cran.r-project.org/package=funHDDC) package installed.

## Running things

Each component was originally developed against a SLURM cluster (see the
`slurm/` folders), since the full simulation study (50 replicates x 3
methods) and the AOMIC cross-validation are too slow to run serially on a
laptop. For local development or a quick smoke test of a single seed/step,
use the top-level `Makefile`:

```bash
make help          # list all available targets
make sim-one        SEED=10   # simulate + cluster one dataset with all 3 methods
make aomic-cv        SEED=5   # run one AOMIC CV replicate locally
```

For the actual paper results, submit the `.sbat` scripts in each component's
`slurm/` folder via `sbatch`, in the order documented in that component's
README. You'll need to fill in `#SBATCH --account=your_account_here` with
your own cluster allocation first.
