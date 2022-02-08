# CeSpGRN: Inferring cell specific GRN using single-cell gene expression data 

[Zhang's Lab](https://xiuweizhang.wordpress.com), Georgia Institute of Technology

Developed by Ziqi Zhang, Jongseok Han

## Description
CeSpGRN is a package that is able to infer cell specific GRN using single-cell gene expression data

* `src` stores the inference algorithms including baseline.
* `test` stores the running script `softODE_test` for Dynamic GRN inference
* `simulator` stores softODE simulator `soft_boolODE` and running script `run_simulator`
* `data` stores the gene expression data
  * sample data is not uploaded because of space limit, and it is available after running simulator in local

## Dependency
```
(required)
pytorch >= 1.15.0 
numpy >= 1.19.5
scipy >= 1.7.1
networkx >= 2.5
sklearn >= 0.24.2

(optional)
matplotlib >= 3.4.3
statsmodels >= 0.12.2
```

## Usage
* Load in the count matrix as a numpy `ndarray`, the matrix should be of the shape `(ncells, ngenes)`. e.g.
  ```python
  import sys, os
  sys.path.append('./src/')
  import numpy as np 

  # load CeSpGRN
  import g_admm as CeSpGRN
  import kernel

  # read in count matrix
  counts = np.load("counts.npy")
  ```
* Set the hyper-parameter including: bandwidth, neighborhoodsize, and lambda. e.g.
  ```python
  # smaller bandwidth means that GRN of cells are more heterogeneous.
  bandwidth = 1
  # number of neighbor being considered when calculating the covariance matrix.
  n_neigh = 30
  # sparsity regulatorization, larger lamb means sparser result.
  lamb = 0.1
  ```
* Calculate the kernel function, and covariance matrix for each cell, e.g.
  ```python
  # calculate PCA of count matrix
  from sklearn.decompose import PCA
  pca_op = PCA(n_components = 10)
  X_pca = pca_op.fit_transform(counts)

  # using X_pca to calculate the kernel function
  K, K_trun = kernel.calc_kernel_neigh(X_pca, k = 5, bandwidth = bandwidth, truncate = True, truncate_param = n_neigh)

  # estimate covariance matrix, output is empir_cov of the shape (ncells, ngenes, ngenes)
  empir_cov = CeSpGRN.est_cov(X = counts, K_trun = K_trun, weighted_kt = True)
  ```
* Estimating cell-specific GRN, e.g.
  ```python
  # estimate cell-specific GRNs, thetas of the shape (ncells, ngenes, ngenes)
  cespgrn = CeSpGRN.G_admm_minibatch(X=counts[:, None, :], K=K, pre_cov=empir_cov, batchsize = 120)
  thetas = cespgrn.train(max_iters=max_iters, n_intervals=100, lamb=lamb)
  ```

An example run is shown in `demo.py`.
