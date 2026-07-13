# this py file simulates data X and A for one setting and one seed.
import os
os.getcwd()
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import statistics
import random
from scipy.stats import norm
import sys

# packages for FDA
from skfda import FDataBasis
from skfda import FDataGrid
from skfda.representation.basis import BSplineBasis
from skfda.representation.basis import TensorBasis
from skfda.misc.covariances import Matern
from skfda.datasets import make_gaussian_process


def matern_simu_only_pure(p=50, n=100, m=100, K=2):
    A = np.full((p, K), 0.0)
    A[0:int(p / 2), 0] = 1
    A[int(p / 2):p, 1] = 1

    tensor_of_factors = np.zeros((K, n, m), float)  # n is the number of samples. Each sample has size K times m.
    # simulating the Z's # should the factors change across n - YES. because we have a covariance matrix for Z as well.

    gp0 = make_gaussian_process(
        n_samples=n, cov=Matern(length_scale=0.1))
    tensor_of_factors[0, :, :] = gp0.data_matrix.squeeze()
    gp1 = make_gaussian_process(
        n_samples=n, cov=Matern())
    tensor_of_factors[1, :, :] = gp1.data_matrix.squeeze()

    tensor_of_errors = np.zeros((p, n, m), float)  # n is the number of samples. Each error sample has size p times m.
    # simulating the E's
    for i in range(p):
        tensor_of_errors[i, :, :] = np.random.multivariate_normal(np.zeros(m), 0.2 * np.eye(m), n)  # has shape n, m.
    # finding the covariance tensor of E's

    tensor_of_X = np.zeros((p, n, m), float)
    for i in range(n):
        tensor_of_X[:, i, :] = A @ tensor_of_factors[:, i, :] + tensor_of_errors[:, i, :]
    return tensor_of_X, A

def matern_simu_only_pure3(p=100, n=250, m=100, K=3):
    A = np.full((p, K), 0.0)
    # randomize the rows corresponding to each cluster
    pure_indices = [0]*34 + [1]*33 + [2]*33
    random.shuffle(pure_indices)
    for i in range(p):
        A[i, pure_indices[i]] = 1

    tensor_of_factors = np.zeros((K, n, m), float)  # n is the number of samples. Each sample has size K times m.
    # simulating the Z's # should the factors change across n - YES. because we have a covariance matrix for Z as well.

    gp0 = make_gaussian_process(
        n_samples=n, cov=Matern(length_scale=0.1))
    tensor_of_factors[0, :, :] = gp0.data_matrix.squeeze()
    gp1 = make_gaussian_process(
        n_samples=n, cov=Matern())
    tensor_of_factors[1, :, :] = gp1.data_matrix.squeeze()
    gp2 = make_gaussian_process(
        n_samples=n, cov=Matern(length_scale=5))
    tensor_of_factors[2, :, :] = gp2.data_matrix.squeeze()

    tensor_of_errors = np.zeros((p, n, m), float)  # n is the number of samples. Each error sample has size p times m.
    # simulating the E's
    for i in range(p):
        sigma_sq_i = np.random.uniform(1,3)
        tensor_of_errors[i, :, :] = np.random.multivariate_normal(np.zeros(
            m), sigma_sq_i * np.eye(m), n)  # has shape n, m.
    # finding the covariance tensor of E's

    tensor_of_X = np.zeros((p, n, m), float)
    for i in range(n):
        tensor_of_X[:, i, :] = A @ tensor_of_factors[:, i, :] + tensor_of_errors[:, i, :]
    return tensor_of_X, A

def matern_simu_aj3(p=60, n=100, m=100, K=3, pure_var_per_cluster = 10):
    A = np.full((p, K), 0.0)
    pos_vars = pure_var_per_cluster//2
    neg_vars = pure_var_per_cluster - pos_vars
    pure_idx = np.random.choice(list(range(p)), K*pure_var_per_cluster, replace=False)
    nonpure_idx = list(set(range(p))-set(pure_idx))
    for k in range(K):
        A[pure_idx[k* pure_var_per_cluster:(k* pure_var_per_cluster)+pos_vars], k] = 1
        A[pure_idx[(k* pure_var_per_cluster)+pos_vars:(k+1)* pure_var_per_cluster], k] = -1

    # generating A_J
    for j in nonpure_idx:
        sj = random.choice(list(range(2, K + 1)))
        supp = random.sample(list(range(K)), sj)
        for k in supp:
            A[j, k] = random.choice([1, -1]) / sj

    tensor_of_factors = np.zeros((K, n, m), float)  # n is the number of samples. Each sample has size K times m.
    # simulating the Z's # should the factors change across n - YES. because we have a covariance matrix for Z as well.

    gp0 = make_gaussian_process(
        n_samples=n, cov=Matern(length_scale=0.1))
    tensor_of_factors[0, :, :] = gp0.data_matrix.squeeze()
    gp1 = make_gaussian_process(
        n_samples=n, cov=Matern(length_scale=1))
    tensor_of_factors[1, :, :] = gp1.data_matrix.squeeze()
    gp2 = make_gaussian_process(
        n_samples=n, cov=Matern(length_scale=5))
    tensor_of_factors[2, :, :] = gp2.data_matrix.squeeze()

    tensor_of_errors = np.zeros((p, n, m), float)
    # simulating the E's
    for i in range(p):
        sigma_sq_i = np.random.uniform(1,3)
        tensor_of_errors[i, :, :] = np.random.multivariate_normal(np.zeros(m), sigma_sq_i * np.eye(m), n)  # has shape n, m.

    tensor_of_X = np.zeros((p, n, m), float)
    for i in range(n):
        tensor_of_X[:, i, :] = A @ tensor_of_factors[:, i, :] + tensor_of_errors[:, i, :]
    return tensor_of_X, A


def kmeans_simu1(p=50, n=100, m=100, K=2):
    """This function simulates data inspired by the Martino et al. (2017) paper.
    To make the simulation amenable to the factor model setup, we simulate
    50-dimensional data, where the columns are iid of the first two dimensions.
    Instead of simply duplicating the first two dimensions, we generate independent samples
    because we want the factor model to have enough variability to capture.
    """
    # generate Fourier basis functions
    s = np.arange(0, 1, 1 / m)
    number_of_directions = 1
    psi = np.zeros((number_of_directions, len(s)), float)
    for j in range(number_of_directions):
        if j % 2 == 0:
            psi[j, :] = np.sqrt(2) * np.sin((j + 2) * np.pi * s)
        else:
            psi[j, :] = np.sqrt(2) * np.cos((j + 1) * np.pi * s)

    # defining the \rho's which control the decay of eigenfunctions
    # rho = np.array([1 / (j + 2) if j < 3 else 1 / (j + 2) ** 2 for j in
    # range(K_tilde)])
    # check that this is the correct size

    # defining the mean functions for the two clusters
    mean_1 = np.zeros((1, m), float)
    mean_1[0, :] = s * (1 - s)
    mean_2 = np.zeros((1, m), float)
    mean_2[0, :] = 4 * (s ** 2) * (1 - s)

    # simulating the data
    X = np.zeros((p, n, m), float)

    # define covariance matrix of scores
    C = np.eye(p)
    for i in range(p//2):
        for j in range(p//2):
            if i != j: C[i, j] = 0.9
    for i in range(p//2, p):
        for j in range(p//2, p):
            if i != j: C[i, j] = 0.9
    # generate the scores
    scores_temp = np.random.multivariate_normal(np.zeros(p),
                                                C, n)
    for sample in range(n):
        for mutlivar_idx in range(p//2):
            X[mutlivar_idx, sample, :] = mean_1[0,:] + scores_temp[sample, mutlivar_idx] * psi[0, :]
        for multivar_idx in range(p//2, p):
            X[multivar_idx, sample, :] = mean_2[0, :] + scores_temp[
                    sample, multivar_idx] * psi[0, :]

    # we want to cluster across the columns for all methods. The k_means
    # algorithm clusters across the first index, so does funHDDC.

    true_A = np.full((p, K), 0.0)
    true_A[0:int(p / 2), 0] = 1
    true_A[int(p / 2):p, 1] = 1
    return X, true_A

def kmeans_simu2(p=50, n=100, m=100, K=2):
    """We extend kmeans_simu1 in this example by adding more directions. The
    coefficients are iid across the different directions as well as across
    the samples.
    """
    # generate Fourier basis functions
    s = np.arange(0, 1, 1 / m)
    number_of_directions = 100
    psi = np.zeros((number_of_directions, len(s)), float)
    for j in range(number_of_directions):
        if j % 2 == 0:
            psi[j, :] = np.sqrt(2) * np.sin((j + 2) * np.pi * s)
        else:
            psi[j, :] = np.sqrt(2) * np.cos((j + 1) * np.pi * s)

    # defining the \rho's which control the decay of eigenfunctions
    sqrt_rho = np.array([1 / math.sqrt(j + 2) if j < 3 else 1 / (j + 2)
                         for j in
    range(number_of_directions)])

    # defining the mean functions for the two clusters
    mean_1 = np.zeros((1, m), float)
    mean_1[0, :] = s * (1 - s)
    mean_2 = mean_1 + np.sum(sqrt_rho[:3].reshape((3),
                                                  1) * psi[:3, :],
                             axis = 0)

    # simulating the data
    X = np.zeros((p, n, m), float)

    # define covariance matrix of scores
    C = np.eye(p)
    for i in range(p//2):
        for j in range(p//2):
            if i != j: C[i, j] = 0.8
    for i in range(p//2, p):
        for j in range(p//2, p):
            if i != j: C[i, j] = 0.8



    for sample in range(n):
        # generate the scores
        scores_temp = np.random.multivariate_normal(np.zeros(p),
                                                    C, number_of_directions)
        for multivar_idx in range(p//2):
            for l in range(number_of_directions):
                X[multivar_idx, sample, :] += scores_temp[l,
                multivar_idx] * sqrt_rho[l] * psi[l, :]
            # add mean
            X[multivar_idx, sample, :] += mean_1[0, :]
        for multivar_idx in range(p//2, p):
            for l in range(number_of_directions):
                X[multivar_idx, sample, :] += scores_temp[l,
                multivar_idx] * sqrt_rho[l] * psi[l, :]
            # add means
            X[multivar_idx, sample, :] += mean_2[0, :]

    # we want to cluster across the columns for all methods. The k_means
    # and funHDDC algorithm clusters across the first index.

    true_A = np.full((p, K), 0.0)
    true_A[0:int(p / 2), 0] = 1
    true_A[int(p / 2):p, 1] = 1
    return X, true_A


number = int(sys.argv[1])
random.seed(number)
sim_type = str(sys.argv[2])
if sim_type=="matern":
    X, true_A = matern_simu_only_pure()
if sim_type == "matern3":
    X, true_A = matern_simu_only_pure3()
if sim_type == "kmeans1":
    X, true_A = kmeans_simu1(n=250)
if sim_type == "kmeans2":
    X, true_A = kmeans_simu2(n=50)
if sim_type == "matern3_aj":
    X, true_A = matern_simu_aj3(p=120, n=250)

data = []

p, n, m =  X.shape
for i in range(p):
    for j in range(n):
        for k in range(m):
            data.append([i, j, k, X[i, j, k]])

# Convert to DataFrame
df = pd.DataFrame(data, columns=['i', 'j', 'k', 'value'])

# Save to CSV
csv_name = "../data/simulation_values_%d.csv"%number
df.to_csv(csv_name, index=False)

# save the true A as well
df = pd.DataFrame(true_A)
A_name = '../data/true_A_%d.csv'%number
df.to_csv(A_name, index=False)
