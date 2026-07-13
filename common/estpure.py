"""This code can be used to replicate the results of the paper titled
ADAPTIVE ESTIMATION IN STRUCTURED FACTOR MODELS WITH APPLICATIONS TO OVERLAPPING CLUSTERING by Bing et al.
It involves an implementation of the first part of the LOVE algorithm where one wants to find K, the number of clusters
and I, a partition of the pure variable set corresponding to different clusters."""

import numpy as np
import math
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import statistics

# helper functions
def check_disjoint(group_list):
    """This function checks if all the sets in the list are disjoint
    Parameters: group_list- a list of sets
    Output: True if all sets are disjoint, otherwise False"""
    if len(group_list) == 1:
        return True
    for i in range(len(group_list) - 1):
        group_i = group_list[i] # this is actually a set
        for j in range(i + 1, len(group_list)):
            group_j = group_list[j]
            if len(group_i.intersection(group_j)) != 0:
                return False
    return True

def union_group(group_list, union):
    """This function does a union operation on all the groups in the list which have common variables
        Parameters: group_list - a list of sets (groups), union - boolean parameter. if True, it forms a new group with the intersections, otherwise merges overlapping groups.
        Output: new list of sets (groups)"""
    new_group = [group_list[0]]
    for i in range(1, len(group_list)):
        group_i = group_list[i]
        in_group_flag = False
        for j in range(len(new_group)):
            group_j = new_group[j]
            if len(group_i.intersection(group_j)) != 0:
                in_group_flag = True
                if union:
                    new_group[j] = group_i.union(group_j)
                else:
                    new_group[j] = group_i.intersection(group_j)
                break
        if not in_group_flag:
            new_group.append(set(group_i))
    return new_group


def pure_var(Sigma, delta=0.02):
    """This function extracts the pure variable partition I and the cluster size K.
    Parameters: Sigma - The estimated covariance matrix, delta- threshold for LOVE algorithm
    Output: pure variable partition I_formatted and the cluster size K"""
    n, p = Sigma.shape
    I = {}  # this is a dictionary
    pure_var = np.array(np.full(p, 1), dtype="bool")
    for i in range(n):
        L = list(range(p))
        L.remove(i)
        row_max = np.max(abs(Sigma[i, L]))
        S_i = [l for l in L if abs(abs(Sigma[i, l]) - row_max) <= 2 * delta]
        for j in S_i:
            L2 = list(range(p))
            L2.remove(j)
            row_max_j = np.max(abs(Sigma[j, L2]))
            if (abs(abs(Sigma[i, j]) - row_max_j)) > 2 * delta:
                pure_var[i] = False
        if pure_var[i]: I[i] = S_i + [i]
        # print(i) - for debugging
    I_formatted = []
    for key, val in I.items():
        # using a set structure here ensures that none of the sets are completely the same, but still we have to ensure that the individual pure rows are not repeated
        if set(val) not in I_formatted: I_formatted.append(set(val))

    # merging clusters that have a common pure row
    while not check_disjoint(I_formatted):
        I_formatted = union_group(I_formatted, True)

    K = len(I_formatted)  # K is the estimated number of clusters
    return I_formatted, K

def sign(a): # helper function that returns the sign of a float variable
    if a > 0 : return 1
    if a < 0: return -1
    return 0

def off_diag_norm(A, B):
    """Helper function that calculates the off-diagonal norm between two diagonal matrices A and B"""
    C = A-B
    n = C.shape[0] # this is the estimated number of pure variables
    for i in range(n):
        C[i,i] = 0.0
    return np.linalg.norm(C)


def cross_val(X, p, n, return_type="delta"):
    """Function to select delta using cross-validation. In the process we also recover A_I and C.
    Parameters: X- matrix of samples from the observed variable X, p- length of each observed vector, n- number of samples,
    return_type can be "delta", which returns the optimal delta or "graph" which produces graphs of the CV loss against each delta"""

    # split X into two halves for cross-validation
    perm = np.arange(n)
    np.random.shuffle(perm)
    X[:] = X[:, perm]

    # we will
    X_hold = X[:, : (n // 2)]
    X2 = X[:, (n // 2):]
    # normalizing the new X's
    mean_X_hold = np.vstack([X_hold.mean(axis=1)] * (n // 2)).T
    Sigma_hold = ((X_hold - mean_X_hold) @ (X_hold - mean_X_hold).T) / (n // 2)

    mean_X2 = np.vstack([X2.mean(axis=1)] * (n // 2)).T
    Sigma = ((X2 - mean_X2) @ (X2 - mean_X2).T) / (n // 2)  # this will be used to construct W

    grid_deltas = np.arange(1.5, 2.5,
                            0.1)  # seq(1.5, 2.5, 0.1) is what the original code from the authors used
    grid_deltas = math.sqrt(math.log(p) / n) * grid_deltas
    losses = np.full(len(grid_deltas), -1.0)
    l = 0
    K_list = np.full(len(grid_deltas), -1)
    for delta in grid_deltas:
        # constructing A_I
        I, K = pure_var(Sigma, delta)
        # print("Number of clusters for tuning parameter", delta, "is :",K)

        lengthI = sum([len(elt) for elt in I])
        A = np.full((p, K), 0.0)
        for a in range(K):
            I_a = sorted(I[a])
            i = I_a[0]
            A[i, a] = 1
            I_a.remove(I_a[0])
            for j in I_a:
                A[j, a] = 1 * sign(Sigma[i, j])
        # extracting A_I from A
        I_flat = [x for xs in I for x in xs]  # flatten I
        A_I = A[I_flat, :]

        # constructing C
        C = np.full((K, K), 0.0)
        for a in range(K):
            for b in range(K):
                if a == b:
                    sum_sigma = 0
                    for i1 in I[a]:
                        for i2 in I[a]:
                            if i1 != i2:
                                sum_sigma += Sigma[i1, i2]
                    C[a, a] = sum_sigma / (len(I[a]) * (len(I[a]) - 1))
                else:
                    sum_sigma = 0
                    for k1 in I[a]:
                        for k2 in I[b]:
                            sum_sigma += A[k1, a] * Sigma[k1, k2] * A[
                                k2, b]  # we note that we are only accessing A_I since k1, k2 are in I
                    C[a, b] = sum_sigma / (len(I[a]) * len(I[b]))
        W = A_I @ C @ A_I.T

        S_II = Sigma_hold[I_flat, :]
        S_II = S_II[:, I_flat]
        if K == 1:
            losses[l] = 1
        else:
            losses[l] = off_diag_norm(W, S_II) / math.sqrt(
                len(I_flat) * (len(I_flat) - 1))  # calculates cross-validation loss
        K_list[l] = K
        l += 1
        # print(losses[l-1])
    if return_type == "delta":
        return grid_deltas[np.argmin(losses)]

    else:
        print("CV loss minimized at", grid_deltas[np.argmin(losses)])
        ax1 = plt.subplot(211)
        ax1.plot(grid_deltas, losses)
        ax1.set_title("CV Loss vs Tuning Param Delta")

        ax2 = plt.subplot(212, sharex=ax1)
        ax2.plot(grid_deltas, K_list)
        ax2.set_title("Estimated K vs Tuning Param Delta")

        plt.show()
        return K_list[np.argmin(losses)]


def simu2(p=20, n=1000, K=2):
    """This function simulates a simple case where we assume that all observed variables are pure variables.
    There are 2 clusters. We assume the first ten are proxies of the first latent variable, next ten are proxies of the second latent variable.
    Parameters: p - size of observed variable X, n - sample size, K - number of clusters
    Output: centered matrix where each column is a sample of the observed variable and the true covariance matrix of X"""

    # constructing Z
    C = np.full((K, K), -1.0)
    for i in range(K):
        C[i, i] = 2 + (i - 1) / 19
    for i in range(K):
        for j in range(K):
            if i != j: C[i, j] = (-1) ** (i + j) * 0.3 ** (abs(i - j)) * min(C[i, i], C[j, j])

    # We assume we only have pure variables
    I = []
    J = list(range(p))
    A = np.full((p, K), 0.0)
    for i in range(p // 2):
        A[i, 0] = 1
        A[10 + i, 1] = 1
    # we assume the first ten are proxies of the first latent variable, next ten are proxies of the second latent variable

    # generate X
    X = np.full((p, n), 0.0)
    sigmas = [1] * p

    for j in range(n):
        Z = np.random.multivariate_normal([0] * K, C, 1).T
        E = np.random.multivariate_normal(np.zeros(p), np.diag(sigmas), 1).T
        X[:, j] = (A @ Z + E).reshape(20)

    # returning centered X
    mean_X = np.vstack([X.mean(axis=1)] * n).T

    return X - mean_X, A @ C @ A.T + np.diag(sigmas)


def simu1(p=400, n=800, K=20):
    """This function simulates data following the procedure in Bing et al section 5.2.
    Parameters: p - size of the observed vector X, n -  sample size, K - number of clusters
    Output: centered matrix where each column is a sample of the observed variable and the true covariance matrix of X"""

    # constructing Z
    C = np.full((K, K), -1.0)
    for i in range(K):
        C[i, i] = 2 + (i - 1) / 19
    for i in range(K):
        for j in range(K):
            if i != j: C[i, j] = (-1) ** (i + j) * 0.3 ** (abs(i - j)) * min(C[i, i], C[j, j])

    # generating A
    # first we construct A_I
    I = []
    J = list(range(p))
    A = np.full((p, K), 0.0)
    # for the convention on configurations, we use: 0: (5,0), 1:(4,1), 2:(3,2), 3:(2,3), 4:(1,4) and so on
    l = list(range(5))
    l = l * int(K / 5)
    for a in range(K):
        pure_var_index = random.sample(J, 5)
        group = random.choice(l)
        l.remove(group)
        for i in pure_var_index:
            A[i, a] = 1
            I.append(i)
            J.remove(i)  # remove this index from the set of impure variables
        negative_index = random.sample(pure_var_index, group % 5)
        for i in negative_index:
            A[i, a] = -1

    # constructing A_J:
    supp = [2, 3, 4, 5]
    for j in J:
        s_j = random.choice(supp)
        supp_j = random.sample(range(K), s_j)
        for k in supp_j:
            A[j, k] = random.choice([-1, 1]) * 1 / s_j

    # generate X
    X = np.full((p, n), 0.0)
    sigma_sqs = np.random.uniform(1.0, 3, size=p)
    for i in range(n):
        E = np.array([np.random.normal(loc=0, scale=math.sqrt(sigmasq)) for sigmasq in sigma_sqs])
        Z = np.random.multivariate_normal([0] * K, C, 1).T
        X[:, i] = (A @ Z + E.reshape(p, 1)).reshape(p)

    # returning centered X
    mean_X = np.vstack([X.mean(axis=1)] * n).T

    return X - mean_X, A @ C @ A.T + np.diag(sigma_sqs)

# function for simulation study
def simulation_study(p, n):
    """This function runs only one simulation of the study in Bing et al.
    We run this 50 times using a py subprocess file to obtain results.
    Parameters: p - size of observed vector X, n - number of samples of X"""
    X, Sigma = simu1(p, n)
    list_deltas = []
    for N in range(50):
        list_deltas.append(cross_val(X, X.shape[0], X.shape[1]))
    optimDelta = statistics.median(list_deltas)
    Sigma_hat = (X @ X.T)/X.shape[1]
    I, K = pure_var(Sigma_hat, optimDelta)
    print(K)

# simulation_study(400, 500) # prints estimated value of K

def simulation_study2(p, n, simutype = "1"):
    '''This is a slightly modified version of the simulation_study function above. This function performs 50 simulations
    for a specified p and n and returns a list of the K (cluster size) achieved from cross-validation for each simulation.
    Parameters: p - size of observed vector X, n - number of samples of X,
    simutype - type of simulation one requires. 1 performs simulations from the function simu1(), 2 performs simulation from the function simu2()
    Output: returns a list of K's obtained for each of the 50 simulations.'''
    nsim = 50
    print("clustering for n =", n)
    K_vals = []
    for sim in tqdm(range(nsim)):
        if simutype==1 : X, Sigma = simu1(p, n)
        else: X, Sigma = simu2(p,n)
        list_deltas = []
        for N in range(50):
            list_deltas.append(cross_val(X, X.shape[0], X.shape[1]))
        optimDelta = statistics.median(list_deltas)
        Sigma_hat = (X @ X.T)/X.shape[1]
        I, K = pure_var(Sigma_hat, optimDelta)
        K_vals.append(K)
        #print ("p =", p, "n =", n_list[i], ": Percentage of Cluster Recovery is", len(np.where(K_vals == 20)[0])/nsim*100, "%" )
    return K_vals