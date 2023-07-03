import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
app_dir = os.path.join(parent_dir, "app")
if app_dir not in sys.path:
      sys.path.append(app_dir)

from utils import config
from utils.StandardScaler import StandardScaler
from pathlib import Path
from os.path import join
import torch as pt
from torch.nn.functional import interpolate
from torch.utils.data import TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



DATA_PATH = Path(os.path.abspath('')) / "data"

# DATASET
#
# Ma: 0.84 - alpha:  1.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0
# Ma: 0.90 - alpha: -2.5, 1.5, 2.5, 4.0, 5.0, 6.0
#


def load_data(filename: str) -> pt.Tensor:
    """Load a file from filename in data folder

    Args:
        filename (string): path to file relative to data folder

    Returns:
        pt.Tensor: Data Tensor
    """
    data = pt.load(join(DATA_PATH, filename))
    return data


def get_coords() -> tuple[pt.Tensor, pt.Tensor]:
    """Read in the x and y coordinates from the coords.pt file and convert the grid to arrays

    Returns:
        tuple[pt.Tensor, pt.Tensor]: x and y coordinates arrays
    """
    # Load the coordinate grid
    print("Loading x, y data")
    coords = load_data("coords.pt")
    x_grid, y_grid = coords["ma0.84_alpha4.00"]

    # reshape 2D grid to 1D array
    x= pt.reshape(x_grid, (x_grid.numel(), 1)).squeeze()
    y= pt.reshape(y_grid, (y_grid.numel(), 1)).squeeze()
    
    return x, y


def make_data_subset(Ma: str = "0.84"):
    """From the original dataset, create a subset with a reduced number of snapshots, for only one Ma number and with interpolated data

    Args:
        Ma (str, optional): Ma number that should be included in the dataset. Defaults to "0.84".
        num_snapshots (int, optional): Number of snapshots that should be included in the dataset. Defaults to 500.
    """
    # load dataset and extract keys
    cp = load_data("cp_clean.pt")
    keys = list(cp.keys())

    for key in keys:
        # if the desired Ma number is not in the key, it will be removed
        if Ma not in key:
            print(key, " will be popped from dict")
            cp.pop(key)

        # otherwise, it will be kept and interpolated
        else:
            print(key, " will be kept and interpolated")
            cp[key] = interpolate_tensor(data_tensor=cp[key][:,:,:config.target_tensor_shape[2]])

    # save the data subset
    print("Saving data subset ...")
    pt.save(cp, join(DATA_PATH, "cp_084_500snaps_interp.pt"))
    print("Done!")


def interpolate_coords():
    """Interpolate the original coordinate grid to a given resolution
    """
    print("INTERPOLATING COORDINATES")

    # load original coordinate grid
    coords = load_data("coords.pt")
    xx, yy = coords[list(coords.keys())[0]]
    print("Original coordinate shape:       ", xx.shape)

    print("Interpolating ...")
    xx_new = interpolate(xx.unsqueeze(0).unsqueeze(0), size=config.target_resolution, mode="bilinear", align_corners=False).squeeze()
    yy_new = interpolate(yy.unsqueeze(0).unsqueeze(0), size=config.target_resolution, mode="bilinear", align_corners=False).squeeze()
    print("Interpolated coordinate shape:   ", xx_new.shape)
    
    print("Saving ...")
    pt.save((xx_new, yy_new), join(DATA_PATH, "coords_interp.pt"))
    print("Done! \n")


def interpolate_tensor(data_tensor: pt.Tensor) -> pt.Tensor:
    """Interpolate a time-resolved data tensor to a given resolution

    Args:
        data_tensor (pt.Tensor): data tensor of shape (y, x, time)

    Returns:
        pt.Tensor: interpolated tensor
    """
    print("INTERPOLATING DATA TENSOR")
    print("Original tensor shape:           ", data_tensor.shape)

    # initialize new tensor with desired shape
    data_tensor_interp = pt.empty(config.target_tensor_shape)

    print("Interpolating ...")
    # loop over each timestep to interpolate the data
    for timestep in range(data_tensor.shape[2]):
        data_tensor_interp[:,:,timestep] = interpolate(data_tensor[:, :, timestep].unsqueeze(0).unsqueeze(0), size=config.target_resolution, mode="bilinear", align_corners=False).squeeze()

    print("Interpolated tensor shape:       ", data_tensor_interp.shape, "\n")
    return data_tensor_interp


def preprocessing():
    """Loads interpolated dataset and coordinates and turns them into TensorDatasets
    """

    # load interpolated dataset and interpolated coords
    data = pt.load(join(DATA_PATH, "cp_084_500snaps_interp.pt"))
    coords = pt.load(join(DATA_PATH, "coords_interp.pt"))
    xx, yy = coords

    # flatten coordinate grids to arrays
    x = xx.flatten(0, 1)
    y = yy.flatten(0, 1)
    print("Grid shape:                  ", xx.shape)
    print("Reshaped grid size:          ", x.shape)

    # split and reshape the data
    train_tensor, val_tensor, test_tensor = split_and_reshape(data, x, y)

    # fit a Standard-scaler on the training data
    print("Fitting Scaler on training data")
    feature_scaler = StandardScaler().fit(train_tensor[:,1:])
    label_scaler = StandardScaler().fit(train_tensor[:,0])

    # scale all tensors and store them in TensorDataset objects
    print("Making TensorDatasets with the scaled features and labels")
    train_dataset = TensorDataset(feature_scaler.scale(train_tensor[:,1:]), label_scaler.scale(train_tensor[:,0]).unsqueeze(-1))
    val_dataset = TensorDataset(feature_scaler.scale(val_tensor[:,1:]), label_scaler.scale(val_tensor[:,0]).unsqueeze(-1))
    test_dataset = TensorDataset(feature_scaler.scale(test_tensor[:,1:]), label_scaler.scale(test_tensor[:,0]).unsqueeze(-1))

    # save all datasets
    print("Saving ...")
    pt.save(train_dataset, join(DATA_PATH, "train_dataset.pt"))
    pt.save(val_dataset, join(DATA_PATH, "val_dataset.pt"))
    pt.save(test_dataset, join(DATA_PATH, "test_dataset.pt"))
    print("Done! \n")


def split_and_reshape(data: pt.Tensor, x: pt.Tensor, y: pt.Tensor) -> tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
    """split the pressure data into train, val and test and reshape them into the appropriate tensor shape
    """
    # identify flow conditions for training and get the split index of the training data for validation
    train_keys = [key for key in list(data.keys()) if key not in config.test_keys]
    split_index = int(config.train_split)

    # initialize train and validation data tensors
    train_data = data[train_keys[0]][:, :, :split_index].flatten(0, 1)
    val_data = data[train_keys[0]][:, :, split_index:].flatten(0, 1)

    # iterate over training flow conditions, split the training data into train and validation and concatenate
    for train_key in train_keys[1:]:
        train_split = data[train_key][:, :, :split_index]
        val_split = data[train_key][:, :, split_index:]
        train_data = pt.concat((train_data, train_split.flatten(0, 1)), dim=1)
        val_data = pt.concat((val_data, val_split.flatten(0, 1)), dim=1)

    # reshape into tensor with the appropriate shape including x, y and timesteps
    print("Shape of train pressure data:        ", train_data.shape)
    train_tensor = reshape_data(x, y, train_data, "train")

    # reshape into tensor with the appropriate shape including x, y and timesteps
    print("Shape of validation pressure data:   ", val_data.shape)
    val_tensor = reshape_data(x, y, val_data, "val")

    # iterate over test flow conditions and concatenate
    test_data = data[config.test_keys[0]].flatten(0, 1)
    for test_key in config.test_keys[1:]:
        test_data = pt.concat((test_data, data[test_key].flatten(0, 1)), dim=1)

    # reshape into tensor with the appropriate shape including x, y and timesteps
    print("Shape of test pressure data:         ", test_data.shape)
    test_tensor = reshape_data(x, y, test_data, "test")

    return train_tensor, val_tensor, test_tensor


def reshape_data(x: pt.Tensor, y: pt.Tensor, pressure_data: pt.Tensor, type: str) -> pt.Tensor:
    """Reshape pressure data and coordinate arrays into a data tensor with timesteps

    Args:
        x (pt.Tensor): x-coordinates flattened
        y (pt.Tensor): y-coordinates flattened
        pressure_data (pt.Tensor): time-resolved pressure data
        type (str): type of tensor regarding training, validation or testing

    Returns:
        pt.Tensor: reshaped n x (cp, x, y, t) tensor
    """
    print("Reshaping ...")
    # initialize parameters for tensor construction
    rows = pressure_data.shape[0] * pressure_data.shape[1]
    cols = 4
    pts_per_timestep = x.shape[0]
    time_steps = pressure_data.shape[1]
    
    # flatten pressure data
    pressure_data_resh = pressure_data.reshape([rows, 1]).squeeze()

    # initialize tensor
    tensor = pt.zeros((rows, cols))

    # assign data to the tensor
    for time_step in range(time_steps):
        start, end = time_step*pts_per_timestep, (time_step+1)*pts_per_timestep
        tensor[start:end, 1] = x
        tensor[start:end, 2] = y
        if type == "train":
            # timesteps ranging from 000-449
            tensor[start:end, 3] = time_step % (config.time_steps_per_cond - config.val_split)
        elif type == "val":
            # timesteps ranging from 450-499
            tensor[start:end, 3] = (time_step + config.train_split) % config.time_steps_per_cond
        else:
            # timesteps ranging from 000-499
            tensor[start:end, 3] = time_step % config.time_steps_per_cond
    tensor[:, 0] = pressure_data_resh

    print("Shape of data tensor:                ", tensor.shape, "\n")
    return tensor

if __name__ == "__main__":
    # interpolate_coords()
    # make_data_subset()
    preprocessing()