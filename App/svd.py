from modules.preprocessing import load_data
import torch as pt
from torch import flatten
from flowtorch.analysis import SVD
import random
import bisect
import matplotlib.pyplot as plt
import matplotlib as mpl
from os.path import join
import numpy as np
random.seed(711)

output_path = "./output"
data_path = "./data"

if __name__ == "__main__":
    # load data and extract keys
    cp_084_data = load_data("cp_084_500snaps.pt")
    keys = list(cp_084_data.keys())

    # sample two random keys for test data except the outer ones
    test_keys = random.sample(keys[1:-1], 2)
    print("The test keys are:       ", test_keys)

    # assemble test data
    X_test = pt.concat((cp_084_data[test_keys[0]].flatten(0, 1), cp_084_data[test_keys[1]].flatten(0, 1)), dim=1)
    print("Shape of test_data is:   ", X_test.shape, "\n")
    
    # extract the train keys and shuffle them, so the alphas are not ordered
    train_keys = [key for key in keys if key not in test_keys]
    random.shuffle(train_keys)
    print("The train keys are:      ", train_keys)

    # assemble train data
    X_train = cp_084_data[train_keys[0]].flatten(0, 1)
    for i in range(1, len(train_keys)):
        X_train = pt.concat((X_train, cp_084_data[train_keys[i]].flatten(0, 1)), dim=1)
    print("Shape of train_data is:  ", X_train.shape, "\n")

    # Apply Singular Value Decomposition with rank truncation
    rank = 2500 
    svd = SVD(X_train - X_train.mean(dim=1).unsqueeze(-1), rank=rank)
    print(svd, "\n")

    # extract the left singular vectors, the right singular vectors and the singular values
    U = svd.U
    s = svd.s
    V_t = svd.V
    print("Shape of U is:           ", U.shape)
    print("Shape of s is:           ", s.shape)
    print("Shape of V is:           ", V_t.shape, "\n")
    pt.save(U, join(data_path, "U.pt"))

    s_sum = s.sum().item()
    # relative contribution
    s_rel = [s_i / s_sum * 100 for s_i in s]
    # cumulative contribution
    s_cum = [s[:n].sum().item() / s_sum * 100 for n in range(s.shape[0])]
    # find out how many singular values we need to reach at least 90 percent
    i_90 = bisect.bisect_right(s_cum, 90)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.bar(range(s.shape[0]), s_rel, align="edge")
    ax2.plot(range(s.shape[0]), s_cum)
    ax2.set_xlim(0, rank)
    ax2.set_ylim(0, 105)
    ax1.set_title("individual contribution in %")
    ax2.set_title("cumulative contribution in %")
    ax2.plot([0, i_90, i_90], [s_cum[i_90], s_cum[i_90], 0], ls="--", color="C3")
    ax2.text(i_90+1, 45, "first {:d} singular values yield {:1.2f}%".format(i_90, s_cum[i_90]))
    plt.savefig(join(output_path, "svd.png"))

    # reconstruct the data matrix
    opt_rank = svd.opt_rank
    print("The optimal rank is:                             ", opt_rank)
    print("Reconstructing X with first ", opt_rank, " POD modes \n")
    X_reconstructed = U[:,:opt_rank] @ pt.diag(s[:opt_rank]) @ V_t[:opt_rank,:]

    # Compute the reconstruction error (MSE)
    reconstruction_error = pt.mean((X_train - (X_reconstructed + X_train.mean(dim=1).unsqueeze(-1))) ** 2).item()
    print("The reconstruction error is (MSE):               ", round(reconstruction_error, 5))

    # Compute the reconstruction accuracy (R-squared)
    total_variance = pt.sum((X_train - pt.mean(X_train)) ** 2)
    residual_variance = pt.sum((X_train - (X_reconstructed + X_train.mean(dim=1).unsqueeze(-1))) ** 2)
    r_squared = 1 - (residual_variance / total_variance).item()
    print("The reconstruction accuracy is (R-squared):      ", round(r_squared, 5), "\n")

    # # Find minimum number of ranks to achieve 90% reconstruction
    # reconstruction_variance = total_variance * 0.9
    # cumulative_variance = np.cumsum(s.numpy()**2)
    # min_rank_truncation = np.argmax(cumulative_variance >= reconstruction_variance.numpy()) + 1

    # print("The rank for 90p reconstruction accuracy is:     ", min_rank_truncation)
    # print("Reconstructing X with first ", min_rank_truncation, " POD modes \n")
    # X_reconstructed = U[:,:min_rank_truncation] @ pt.diag(s[:min_rank_truncation]) @ V_t[:min_rank_truncation,:]

    # # Compute the reconstruction accuracy (R-squared)
    # total_variance = pt.sum((X_train - pt.mean(X_train)) ** 2)
    # residual_variance = pt.sum((X_train - (X_reconstructed + X_train.mean(dim=1).unsqueeze(-1))) ** 2)
    # r_squared = 1 - (residual_variance / total_variance).item()
    # print("The reconstruction accuracy is (R-squared):      ", round(r_squared, 5), "\n")

