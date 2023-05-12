from modules.preprocessing import load_data
import torch as pt
from torch import flatten
import flowtorch as ft
import random
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
    

