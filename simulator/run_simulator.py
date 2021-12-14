# In[0]
import sys, os

sys.path.append('./')
import simulator_soft_ODE as simulator
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.decomposition import PCA
import numpy as np

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
    libsize = np.median(np.sum(counts, axis=1))
    counts = counts / np.sum(counts, axis=1)[:, None] * libsize
    counts = np.log1p(counts)
    return counts

import importlib
importlib.reload(simulator)


# In[1] simulate bifurcating data where the graph in/out degrees kept
path = "../../data/simulation_data/"
sergio_path = "./initial_grn/Interaction_cID_8"

mgenes = 200
ntimes = 200
traj = "bifurc"

for seed in range(5):
    for stepsize in [0.001]:
        for ncells in [1000]: ## (stepsize = 0.0003, mgenes = 200) - bifurcating
            for (ngenes, ntfs) in [(20, 5)]:

                # NEED TO "KEEP DEGREE = FALSE" IN SERGIO INITIAL GRAPH
                G0 = simulator.load_sub_sergio(grn_init = sergio_path, sub_size = ngenes, ntfs = ntfs, init_size = 100)

                print("Load SERGIO")
                simu_setting = {"ncells": ncells, # number of cells
                                "ntimes": ntimes, # time length for euler simulation
                                "integration_step_size": stepsize, # stepsize for each euler step
                                # parameter for dyn_GRN
                                "ngenes": ngenes, # number of genes 
                                "mode": "TF-TF&target", # mode of the simulation, `TF-TF&target' or `TF-target'
                                "ntfs": ntfs,  # number of TFs
                                # nchanges also drive the trajectory, but even if don't change nchanges, there is still linear trajectory
                                "nchanges": 4, # number of changing edges for each interval (ngenes 10 > nchanges 2 / ngenes 20 > nchanges 4 )
                                "change_stepsize": 20, # number of stepsizes for each change
                                "density": 0.15, # number of edges
                                "seed": seed, # random seed
                                "dW": None,
                                # the changing point must be divided exactly by the change_stepsize, or there will be issue.
                                "backbone": np.array(["0_1"] * 80 + ["1_2"] * 60 + ["1_3"] * 60),
                                # "backbone": np.array(["0_1"] * ntimes),
                                "keep_degree": False,
                                "mgenes": mgenes,
                                "G0": G0
                                }
                data = str(traj) + "ngenes_" + str(ngenes) + "_ncell_" + str(ncells) + "_seed_" + str(seed)
                if not os.path.exists(path + data):
                    os.makedirs(path + data)
                print("Simulation Start")
                results = simulator.run_simulator(**simu_setting)

                fig = plt.figure(figsize = (10, 7))
                ax = fig.add_subplot()
                umap_op = UMAP(n_components = 2)
                pca_op = PCA(n_components = 2)

                X = results["true count"].T
                pt = results["pseudotime"]
                GRNs = results["GRNs"]
                # make the graph symmetric, originally only one direction, so we can simply sum them up

                # don't calculate diagonal elment twice
                diag_values = np.concatenate([np.diag(np.diag(GRNs[x, :, :]))[None, :, :] for x in range(GRNs.shape[0])], axis = 0)
                GRNs = GRNs + np.transpose(GRNs, (0, 2, 1)) - diag_values
                print("Simulation Preprocessing")
                X = preprocess(X)
                X_umap = pca_op.fit_transform(X)
                ax.scatter(X_umap[:, 0], X_umap[:, 1], s = 5, c = pt)
                fig.savefig(path + data + "/true_count_plot.png", bbox_inches = "tight")

                fig = plt.figure(figsize = (10, 7))
                ax = fig.add_subplot()
                X = results["observed count"].T
                pt = results["pseudotime"]
                X = preprocess(X)
                X_umap = pca_op.fit_transform(X)
                ax.scatter(X_umap[:, 0], X_umap[:, 1], s = 5, c = pt)
                fig.savefig(path + data + "/observed_count_plot.png", bbox_inches = "tight")

                np.save(path + data + "/true_count.npy", results["true count"].T)
                np.save(path + data + "/obs_count.npy", results["observed count"].T)
                np.save(path + data + "/pseudotime.npy", results["pseudotime"])
                np.save(path + data + "/backbone.npy", results["backbone"])
                np.save(path + data + "/GRNs.npy", GRNs)
                

# %%
