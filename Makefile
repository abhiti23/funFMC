.PHONY: setup \
        sim-data sim-fm sim-kmeans sim-funhddc sim-evaluate sim-one \
        clt-v2 clt-plugin \
        aomic-cv aomic-aggregate aomic-final aomic-plots \
        clean help

# A single seed used by the "smoke test" targets below (sim-one, aomic-cv).
# Override on the command line, e.g. `make sim-one SEED=20`.
SEED ?= 10

help:
	@echo "Targets:"
	@echo "  setup           install Python dependencies from requirements.txt"
	@echo ""
	@echo "  Simulation study (run locally for one seed; use slurm/*.sbat for the full 50-seed sweep):"
	@echo "  sim-data        simulate one dataset  (SEED=$(SEED))"
	@echo "  sim-fm          run our method (funFMC.py) on that dataset"
	@echo "  sim-kmeans      run functional k-means on that dataset"
	@echo "  sim-funhddc     run funHDDC (R) on that dataset"
	@echo "  sim-one         run sim-data + all three clustering methods for SEED"
	@echo "  sim-evaluate    aggregate metrics across all simulated datasets in results/"
	@echo ""
	@echo "  Asymptotic normality:"
	@echo "  clt-v2          run the oracle-covariance CLT check"
	@echo "  clt-plugin      run the plug-in standard-error CLT check"
	@echo ""
	@echo "  Data analysis (assumes data-analysis/data/aomic_data_200.npy is present):"
	@echo "  aomic-cv        run one CV replicate for the AOMIC data  (SEED=$(SEED))"
	@echo "  aomic-aggregate concatenate results/delta_*.txt into results/all_deltas.txt"
	@echo "  aomic-final     run final_clus.py (needs all_deltas.txt to exist first)"
	@echo "  aomic-plots     produce the Yeo-17 and cortical-surface figures"
	@echo ""
	@echo "  clean           remove generated results (keeps .gitkeep placeholders)"

setup:
	pip install -r requirements.txt --break-system-packages
	@echo "Note: the simulation study's funHDDC step also needs R with the 'funHDDC' package installed."

# --------------------------------------------------------------------
# Simulation study
# --------------------------------------------------------------------
sim-data:
	cd simulation-study/python && python3 simulate_data.py $(SEED) matern3_aj

sim-fm:
	cd simulation-study/python && python3 funFMC.py $(SEED)

sim-kmeans:
	cd simulation-study/python && python3 kmeans_func_clus.py $(SEED)

sim-funhddc:
	cd simulation-study/R && Rscript funHDDC_clus.R $(SEED)

sim-one: sim-data sim-fm sim-kmeans sim-funhddc
	@echo "Ran seed $(SEED) through simulation + all three clustering methods."

sim-evaluate:
	cd simulation-study/python && python3 evaluate.py

# --------------------------------------------------------------------
# Asymptotic normality
# --------------------------------------------------------------------
clt-v2:
	cd asymptotic-normality && python3 verify_clt_v2.py

clt-plugin:
	cd asymptotic-normality && python3 verify_clt_plugin.py

# --------------------------------------------------------------------
# Data analysis (AOMIC)
# --------------------------------------------------------------------
aomic-cv:
	cd data-analysis/python && python3 one_cv_gl.py $(SEED)

aomic-aggregate:
	cat data-analysis/results/delta_*.txt > data-analysis/results/all_deltas.txt

aomic-final:
	cd data-analysis/python && python3 final_clus.py

aomic-plots:
	cd data-analysis/python && python3 plot_yeo17_network_loadings.py
	cd data-analysis/python && python3 plot_surface_clusters.py

# --------------------------------------------------------------------
clean:
	find simulation-study/data simulation-study/results \
	     data-analysis/data data-analysis/results \
	     asymptotic-normality/results \
	     -type f ! -name '.gitkeep' -delete
