from os import makedirs
from os.path import join
import torch as pt
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt

data_path = "./data"

# DATASET
# 13 combinations of Ma and alpha:
#
# Ma: 0.84 - alpha:  1.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0
# Ma: 0.90 - alpha: -2.5, 1.5, 2.5, 4.0, 5.0, 6.0
#
#


def load_data(filename):
    data = pt.load(join(data_path, filename))
    return data


def reshape_data(x: pt.Tensor, y: pt.Tensor, cp: dict) -> pt.Tensor:

    print("initial cp_shape: ", cp.shape)
    #x = np.tile(x, (cp.shape[2]))
    #y = np.tile(y, (cp.shape[2]))
    ma = pt.full((int(cp.numel()/2000),), 0.84)
    alpha = pt.full((int(cp.numel()/2000),), 4.00)
    time_step = pt.full((int(cp.numel()/2000),), 1)
    cp = cp[:, :, 0]
    cp = pt.reshape(cp, (cp.numel(), 1)).squeeze()
    
    
    print("shape of x       = ", x.shape)
    print("shape of y       = ", y.shape)
    print("shape of ma      = ", ma.shape)
    print("shape of alpha   = ", alpha.shape)
    print("shape of cp_test = ", cp.shape)

    data = pt.zeros((cp.numel(), 6))
    data[:, 0] = x
    data[:, 1] = y
    data[:, 2] = ma
    data[:, 3] = alpha
    data[:, 4] = time_step
    data[:, 5] = cp

    print("Dataset length =", data.shape)
    return data


def get_coords():
    coords = load_data("coords.pt")
    x_grid, y_grid = coords["ma0.84_alpha4.00"]

    x= pt.reshape(x_grid, (x_grid.numel(), 1)).squeeze()
    y= pt.reshape(y_grid, (y_grid.numel(), 1)).squeeze()
    
    return x, y

def get_cp():
    return


if __name__ == "__main__":
    x, y = get_coords()
    cp_test = load_data("cp_ma0.84_alpha4.00.pt")
    
    data = reshape_data(x, y, cp_test)

    dataset = pt.utils.data.TensorDataset(data[:, :5], data[:, 5])
    print(dataset[5000])