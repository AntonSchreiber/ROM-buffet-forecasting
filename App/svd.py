from modules.preprocessing import load_data
import torch as pt
from torch import flatten
from flowtorch.analysis import SVD
import random
import bisect
import matplotlib.pyplot as plt
import matplotlib as mpl
random.seed(711)

if __name__ == "__main__":
    # load data and extract keys
    cp_084_data = load_data("cp_084_500snaps.pt")
    keys = list(cp_084_data.keys())

    # sample two random keys for test data except the outer ones
    test_keys = random.sample(keys[1:-1], 2)
    print("The test keys are:       ", test_keys)

    # assemble test data
    X_test = pt.concat((cp_084_data[test_keys[0]].flatten(0, 1), cp_084_data[test_keys[1]].flatten(0, 1)), dim=0)
    print("Shape of test_data is:   ", X_test.shape)
    
    # extract the train keys and shuffle them, so the alphas are not ordered
    train_keys = [key for key in keys if key not in test_keys]
    random.shuffle(train_keys)
    print("The train keys are:      ", train_keys)

    # assemble train data
    X_train = cp_084_data[train_keys[0]].flatten(0, 1)
    for i in range(1, len(train_keys)):
        X_train = pt.concat((X_train, cp_084_data[train_keys[i]].flatten(0, 1)), dim=0)
    print("Shape of train_data is:  ", X_train.shape)

    # Apply Singular Value Decomposition
    rank = 58
    svd = SVD(X_train, rank=rank)
    print(svd)

    s = svd.s
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
    plt.show()


    

