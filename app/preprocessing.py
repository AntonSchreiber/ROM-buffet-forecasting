import os
from os.path import join
import sys
from pathlib import Path

# include app directory into sys.path
REMOTE= False
parent_dir = Path(os.path.abspath('')).parent if REMOTE else Path(os.path.abspath(''))
app_dir = join(parent_dir, "app")
if app_dir not in sys.path:
      sys.path.append(app_dir)

import utils.config as config
from utils.Scaler import MinMaxScaler_1_1
from utils.AutoencoderDataset import AutoencoderDataset
import torch as pt
from torch.nn.functional import interpolate
import random
random.seed(10)

DATA_PATH = join(parent_dir, "data")

# DATASET
# Ma: 0.84 - alpha:  1.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0


def load_data(filename: str) -> pt.Tensor:
    """Load a file from filename in data folder

    Args:
        filename (string): path to file relative to data folder

    Returns:
        pt.Tensor: Data Tensor
    """
    data = pt.load(join(DATA_PATH, filename))
    return data


def get_coords() -> tuple:
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


def svd_preprocesing():
    """Loads interpolated dataset and wraps it into appropriate tensors for SVD computations
    """
    print("Creating SVD datasets from surface pressure data ...")
    # load data and extract keys
    data = pt.load(join(DATA_PATH, "cp_084_500snaps_interp.pt"))
    keys = list(data.keys())

    # sample two random keys for test data except the outer ones
    test_keys = config.test_keys
    print("The test keys are:       ", test_keys)

    # assemble test data
    X_test_1 = data[test_keys[0]].flatten(0, 1)
    X_test_2 = data[test_keys[1]].flatten(0, 1)
    print("Shape of test_data_1 is:   ", X_test_1.shape, "\n")

    # extract the train keys and shuffle them
    train_keys = [key for key in keys if key not in test_keys]
    random.shuffle(train_keys)
    print("The train keys are:      ", train_keys)

    # assemble train data
    X_train = data[train_keys[0]].flatten(0, 1)
    for i in range(1, len(train_keys)):
        X_train = pt.concat((X_train, data[train_keys[i]].flatten(0, 1)), dim=1)
    print("Shape of train_data is:  ", X_train.shape, "\n")

    # # center datasets by temporal mean
    # X_train_centered = X_train- X_train.mean(dim=1).unsqueeze(-1)
    # X_test_1_centered = X_test_1 - X_test_1.mean(dim=1).unsqueeze(-1)
    # X_test_2_centered = X_test_2 - X_test_2_centered.mean(dim=1).unsqueeze(-1)

    # fit a Min-Max-Scaler on the training data
    print("Fitting Scaler on training data")
    scaler = MinMaxScaler_1_1().fit(X_train)
    # scaler_centered = MinMaxScaler_1_1().fit(X_train_centered)

    # save all datasets
    print("Saving ...")
    pt.save(scaler.scale(X_train), join(DATA_PATH, "SVD", "X_train.pt"))
    pt.save(scaler.scale(X_test_1), join(DATA_PATH, "SVD", "X_test_1.pt"))
    pt.save(scaler.scale(X_test_2), join(DATA_PATH, "SVD", "X_test_2.pt"))
    # pt.save(scaler_centered.scale(X_train_centered), join(DATA_PATH, "SVD", "X_train_centered.pt"))
    # pt.save(scaler_centered.scale(X_test_1_centered), join(DATA_PATH, "SVD", "X_test_1_centered.pt"))
    # pt.save(scaler_centered.scale(X_test_2_centered), join(DATA_PATH, "SVD", "X_test_2_centered.pt"))
    print("Done! \n")


def autoencoder_preprocessing():
    """Loads interpolated dataset and wraps it into TensorDatasets for autoencoder training
    """
    print("Creating custom autoencoder datasets from surface pressure data ...")

    # load interpolated dataset
    data = pt.load(join(DATA_PATH, "cp_084_500snaps_interp.pt"))

    # split and reshape the data
    train_cp, val_cp, test_cp = split_data_all(data)

    # fit a Min-Max-Scaler on the training data
    print("Fitting Scaler on training data")
    cp_scaler = MinMaxScaler_1_1().fit(train_cp)

    # scale all tensors and create custom Datasets
    print("Making AutoencoderDatasets with the scaled cp")
    train_dataset = AutoencoderDataset(cp_scaler.scale(train_cp))
    val_dataset = AutoencoderDataset(cp_scaler.scale(val_cp))
    test_dataset = AutoencoderDataset(cp_scaler.scale(test_cp))    

    # save all datasets
    print("Saving ...")
    pt.save(train_dataset, join(DATA_PATH, "VAE", "train_dataset.pt"))
    pt.save(val_dataset, join(DATA_PATH, "VAE", "val_dataset.pt"))
    pt.save(test_dataset, join(DATA_PATH, "VAE", "test_dataset.pt"))
    print("Done! \n")


def split_data_all(data: pt.Tensor) -> tuple:
    """split the pressure data into train, val and test
    """

    # identify flow conditions for training and get the split index of the training data for validation
    train_keys = [key for key in list(data.keys()) if key not in config.test_keys]
    split_index = int(config.train_split)

    # initialize train and validation data tensors
    if config.mini_dataset:
        train_cp = data[train_keys[0]][:, :, :config.mini_train_per_cond]
        val_cp = data[train_keys[0]][:, :, split_index:(split_index + config.mini_val_per_cond)]
    else:
        train_cp = data[train_keys[0]][:, :, :split_index]
        val_cp = data[train_keys[0]][:, :, split_index:]

    # iterate over training flow conditions, split the training data into train and validation and concatenate
    for train_key in train_keys[1:]:
        train_split = data[train_key][:, :, :split_index]
        val_split = data[train_key][:, :, split_index:]
        if config.mini_dataset:
            train_cp = pt.concat((train_cp, train_split[:, :, :config.mini_train_per_cond]), dim=2)
            val_cp = pt.concat((val_cp, val_split[:, :, :config.mini_val_per_cond]), dim=2)
        else:
            train_cp = pt.concat((train_cp, train_split), dim=2)
            val_cp = pt.concat((val_cp, val_split), dim=2)

    print("Shape of training cp:    ", train_cp.shape)
    print("Shape of val cp:         ", val_cp.shape)

    # iterate over test flow conditions and concatenate
    if config.mini_dataset:
        test_cp = data[config.test_keys[0]][:, :, :config.mini_test_per_cond]
    else:
        test_cp = data[config.test_keys[0]]

    for test_key in config.test_keys[1:]:
        if config.mini_dataset:
            test_cp = pt.concat((test_cp, data[test_key][:, :, :config.mini_test_per_cond]), dim=2)
        else:
            test_cp = pt.concat((test_cp, data[test_key]), dim=2)
    
    print("Shape of test cp:        ", test_cp.shape)

    return train_cp, val_cp, test_cp


def single_flow_cond_preprocessing():
    """ preprocessing for the single flow condition pipeline """
    print("Creating SVD and VAE datasets for the single flow condition training pipeline ...")
    # load interpolated dataset and pick a flow condition
    data = pt.load(join(DATA_PATH, "cp_084_500snaps_interp.pt"))
    flow_cond = list(data.keys())[3]
    data = data[flow_cond]
    data.shape
    print("Flow condtion:       " ,flow_cond, "\n")

    # split and reshape the data
    train_cp, test_cp = split_data_single(data)

    # fit a Standard-scaler on the training data
    print("Fitting Scaler on training data \n")
    cp_scaler = MinMaxScaler_1_1().fit(train_cp)

    # scale tensors and create custom VAE datasets
    print("Scaling datasets ...")
    train_cp = cp_scaler.scale(train_cp)
    test_cp = cp_scaler.scale(test_cp)
    print("Shape of train:      ", train_cp.shape)
    print("Shape of test:       ", test_cp.shape, "\n")

    # save all datasets
    print("Saving ...")
    pt.save(train_cp, join(DATA_PATH, "pipeline_single", "train_dataset.pt"))
    pt.save(test_cp, join(DATA_PATH, "pipeline_single", "test_dataset.pt"))
    print("Done! \n")


def split_data_single(data):
    """ split single flow cond of dataset into train and test"""
    num_train = int(data.shape[2] * config.single_flow_cond_train_share)
    print("Number of train samples:     ", num_train)
    print("Number of test samples:      ", data.shape[2] - num_train)

    return data[:, :, :num_train], data[:, :, num_train:]


def split_data_multi():
    """ split dataset into multiple flow conds for train (&val) and a single flow cond for test"""
    return


if __name__ == "__main__":
    # interpolate_coords()
    # make_data_subset()
    svd_preprocesing()
    # autoencoder_preprocessing()
    # single_flow_cond_preprocessing()