# In[1]
import pandas as pd
import numpy as np
import torch
import torch.nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sys, os
sys.path.append('../src/')
from scipy.spatial.distance import pdist, squareform

import bmk_beeline as bmk
import genie3, g_admm
import kernel
import time
import gc
from sklearn.decomposition import PCA
from umap import UMAP

# In[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams["font.size"] = 20

def preprocess(counts): 
    """\
    Input:
    counts = (ntimes, ngenes)
    
    Description:
    ------------
    Preprocess the dataset
    """
    # normalize according to the library size
    
    libsize = np.median(np.sum(counts, axis = 1))
    counts = counts / np.sum(counts, axis = 1)[:,None] * libsize
        
    counts = np.log1p(counts)
    return counts

pca_op = PCA(n_components = 10)


def cal_cov(X, K_trun):
    start_time = time.time()
    X = torch.FloatTensor(X)
    ntimes = X.shape[0]
    ngenes = X.shape[1]

    for t in range(ntimes * nsamples):
        weight = torch.FloatTensor(K_trun[t, :])
        # assert torch.sum(weight) == 1

        bin_weight = torch.FloatTensor((K_trun[t, :] > 0).astype(np.int))
        sample_mean = torch.sum(X * weight[:, None], dim = 0)

        norm_sample = X - sample_mean[None, :]
        empir_cov[t] = torch.sum(torch.bmm(norm_sample[:,:,None], norm_sample[:,None,:]) * weight[:,None, None], dim=0)
    print("time calculating the kernel function: {:.2f} sec".format(time.time() - start_time))
    return empir_cov


# In[3]
seeds = [eval(sys.argv[1])]
interval = 20
ntimes = 1000
nsamples = 1
max_iters = 500
trajs = ["linear", "bifurc"]
path = "../../data/simulation_data/"

for kt in [True]:
    for traj in trajs:
        for (ngenes, ntfs) in [(20, 5)]:
            for seed in seeds:
                result_dir = "../results_softODE_hyperparameter/" + str(traj) + "_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
                
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                
                data_dir = path + str(traj) +  "_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
                X = np.load(data_dir + "true_count.npy")
                sim_time = np.load(data_dir + "pseudotime.npy")

                pca_op = PCA(n_components = 20)
                umap_op = UMAP(n_components = 2, min_dist = 0.8)
                X_pca = pca_op.fit_transform(X)
                X_umap = umap_op.fit_transform(X)

                fig = plt.figure(figsize  = (10,7))
                ax = fig.add_subplot()
                ax.scatter(X_pca[:, 0], X_pca[:, 1], c = sim_time)

                ax.set_xlabel("PCA1")
                ax.set_ylabel("PCA2")
                ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
                fig.savefig(result_dir + "X_pca.png", bbox_inches = "tight")

                fig = plt.figure(figsize  = (10,7))
                ax = fig.add_subplot()
                ax.scatter(X_umap[:, 0], X_umap[:, 1], c = sim_time)

                ax.set_xlabel("UMAP1")
                ax.set_ylabel("UMAP2")
                ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
                fig.savefig(result_dir + "X_umap.png", bbox_inches = "tight")


                for bandwidth in [10]:
                    for truncate_param in [30]:
                        empir_cov = torch.zeros(ntimes * nsamples, ngenes, ngenes)
                        # calculate the kernel function, didn't use dimension reduction
                        K, K_trun = kernel.calc_kernel_neigh(X_pca, k = 5, bandwidth = bandwidth, truncate = True, truncate_param = truncate_param)

                        # plot kernel function
                        fig = plt.figure(figsize = (20, 7))
                        axs = fig.subplots(1, 2)
                        axs[0].plot(K[int(ntimes * nsamples/2), :])
                        axs[1].plot(K_trun[int(ntimes * nsamples/2), :])
                        fig.suptitle("kernel_" + str(bandwidth) + "_" + str(truncate_param))
                        fig.savefig(result_dir + "kernel_" + str(bandwidth) + "_" + str(truncate_param) + ".png", bbox_inches = "tight")
                        print("number of neighbor being considered: " + str(np.sum(K_trun[int(ntimes * nsamples/2), :] > 0)))
                        # building weighted covariance matrix, output is empir_cov of the shape (ntimes * nsamples, ngenes, ngenes)
                        if kt:
                            empir_cov = g_admm.est_cov(X = X, K_trun = K_trun, weighted_kt = True)
                        else:
                            empir_cov = cal_cov(X = X, K_trun = K_trun)

                        # run the model
                        for lamb in [0.001]:
                            start_time = time.time() 
                            # test model without TF
                            thetas = np.zeros((ntimes * nsamples,ngenes,ngenes))

                            # setting from the paper over-relaxation model
                            alpha = 2
                            rho = 1.7
                            gadmm_batch = g_admm.G_admm_minibatch(X=X[:, None, :], K=K, pre_cov=empir_cov, batchsize = 100)
                            thetas = gadmm_batch.train(max_iters=max_iters, n_intervals=100, alpha=alpha, lamb=lamb, rho=rho, theta_init_offset=0.1)
                            if kt:
                                np.save(file = result_dir + "thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_kt.npy", arr = thetas) 
                            else:
                                np.save(file = result_dir + "thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) +  ".npy", arr = thetas) 
                            
                            print("time calculating thetas: {:.2f} sec".format(time.time() - start_time))
                            del thetas
                            gadmm_batch = None
                            gc.collect()

# %%