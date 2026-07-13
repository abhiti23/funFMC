library(funHDDC)

# Get command-line arguments
args <- commandArgs(trailingOnly = TRUE)
seed_number <- as.integer(args[1])
sim_name <- paste0("../data/simulation_values_",seed_number,".csv")
df <- read.csv(sim_name)
p <- max(df$i) + 1
n <- max(df$j) + 1
m <- max(df$k) + 1
X <- array(0, dim = c(p, n, m))

for (row in 1:nrow(df)) {
    i <- df$i[row] + 1  # R is 1-indexed
    j <- df$j[row] + 1
    k <- df$k[row] + 1
    X[i, j, k] <- df$value[row]
}

# converting raw data into a functional data object using a b-spline basis with 10 coefficients
basis<- create.bspline.basis(c(0,1), nbasis=10)
for (i in 1:n){
  variable_name <- paste0("var",i)
  temp_fd <- smooth.basis(argvals=seq(0,1,length.out=100), y=t(X[,i,]), basis)$fd
  assign(variable_name, temp_fd)
}

# the function funHDDC takes a list of univariate fd objects with number of samples in each being the number of observations to be clustered.
l = list()
for (i in 1:n) {
  var_name <- paste0("var", i)
  l[[i]] <- get(var_name)
}
names(l) <- paste0("var", 1:n)

# Model specifications ordered from most to least complex.
# funHDDC supports: "AkjBkQkDk", "AkBkQkDk", "ABkQkDk", "AkjBQkDk",
#                   "AkBQkDk", "ABQkDk", "AkjBkQkD", "AkBkQkD",
#                   "ABkQkD",  "AkjBQkD",  "AkBQkD",  "ABQkD"
model_specs <- c("AkjBkQkDk", "AkBkQkDk", "ABkQkDk", "AkjBQkDk")

res.multi <- NULL
successful_model <- NULL

for (model in model_specs) {
  cat(sprintf("Trying model specification: %s\n", model))

  tryCatch({
    result <- funHDDC(l, K=3, model=model, init="kmeans", threshold=0.2)

    # Check for divergence: funHDDC divergence often shows as NA/NaN in
    # log-likelihood or degenerate cluster assignments (all in one cluster)
    has_na        <- any(is.na(result$loglik)) || any(is.nan(result$loglik))

    if (has_na) {
      cat(sprintf("  Model %s produced degenerate results (NA loglik), skipping.\n", model))
    } else {
      cat(sprintf("  Model %s converged successfully.\n", model))
      res.multi        <- result
      successful_model <- model
      break
    }

  }, error = function(e) {
    cat(sprintf("  Model %s failed with error: %s\n", model, conditionMessage(e)))
  }, warning = function(w) {
    cat(sprintf("  Model %s raised a warning: %s\n", model, conditionMessage(w)))
  })
}

write.table(res.multi$class, paste0("../results/funHDDC_result_",seed_number,".txt"), sep = "\t", row.names = FALSE, col.names = FALSE)
