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
TARGET_SHAPE_SLICE = (256, 128)
TARGET_SHAPE_TENSOR = (256, 128, 500)

# DATASET
# 13 combinations of Ma and alpha:
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


def reshape_data_depr(x: pt.Tensor, y: pt.Tensor, cp: dict, cases: dict) -> pt.Tensor:  
    """Reshapes all of the data into the desired shape for the models

    Args:
        x (pt.Tensor): x-coordinates array
        y (pt.Tensor): y-coordinates array
        cp (dict): time dependant pressure data
        cases (dict): dict with the case names as keys and the filtered Ma and alpha as vals

    Returns:
        pt.Tensor: Fully assembled data tensor
    """
    print("Reshaping data")
    # Extract information about the data dimesnions
    dataset_len = sum([val.numel() for _, val in cp.items()])
    keys = list(cp.keys())
    vals_per_step = cp[keys[0]].shape[0]*cp[keys[0]].shape[1]
    num_timesteps = cp[keys[0]].shape[2]

    # clone x and y by the amount of timesteps to achieve desired dimension
    x = np.tile(x, (num_timesteps,))
    y = np.tile(y, (num_timesteps,))

    # creating array with timesteps 
    t = np.zeros((cp[keys[0]].numel(),))
    for step in range(1, num_timesteps):
        t1, t2= step*vals_per_step, (step+1)*vals_per_step
        t[t1:t2] = step

    # Create the data tensor in 16bit float format
    data = np.zeros((dataset_len, 6), dtype=np.float32)

    # Loop over Ma-alpha confs
    for i, (key, vals) in enumerate(cp.items()):
        # start and end indices for current conf in data tensor
        start, end = i*vals.numel(), (i+1)*vals.numel()
   
        # assigning data to corresponding tensor column
        data[start:end, 0] = x
        data[start:end, 1] = y
        data[start:end, 2] = np.full((vals.numel(),), cases[key][0])
        data[start:end, 3] = np.full((vals.numel(),), cases[key][1])
        data[start:end, 4] = t
        data[start:end, 5] = np.reshape(vals, (vals.numel(), 1)).squeeze()

    print("Dataset shape     =", data.shape)
    print("Dataset size      =", data.nbytes /1e+9, "GB \n")

    print("Data reshaped")
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

def get_cp_and_cases(cp_filename: str) -> tuple[pt.Tensor, dict]:
    """Load the raw cp data and extract the Ma and alpha

    Returns:
        tuple[pt.Tensor, dict]: cp tensor and dict with the case names as keys and the filtered Ma and alpha as vals
    """
    print("Loading cp data")
    # Load the cp data
    cp = load_data(cp_filename)

    # Get the conf names
    keys = list(cp.keys())

    # discard 75% of the snapshots for now
    cp[keys[0]], _ = cp[keys[0]].split([500, 1500], dim=2)
    print(cp[keys[0]].shape)
    cp[keys[1]], _ = cp[keys[1]].split([500, 1500], dim=2)
    print(cp[keys[1]].shape)

    # Delete the "ma" and "_alpha" from the keys and convert to float
    cases = [key.replace("ma", "").replace("_alpha", "") for key in list(cp.keys())]
    cases_dict = {keys[i]: [float(case[:4]), float(case[4:])] for i, case in enumerate(cases)}

    return cp, cases_dict


def split_scale_save(df: pd.DataFrame, train_size: float, val_size: float, test_size: float):
    """Split the data into training, validation and testing data

    Args:
        df (pd.DataFrame): Dataframe
        train_size (float): Size of training data
        val_size (float): Size of validation data
        test_size (float): Size of testing data
    """
    # Split the data into training, validation and testing data
    val_end = train_size + val_size
    n = len(df)

    print("Splitting data")
    df_train = df[0:int(train_size*n)]
    df_val = df[int(train_size*n):int(val_end*n)]
    df_test = df[int(val_end*n):]

    # Train MinMaxScaler on training data
    print("Fitting MinMaxScaler on training data")
    scaler = MinMaxScaler()
    scaler.fit(df_train)

    # Scale subsets with the fitted scaler
    print("Scaling datasets with the fitted scaler")
    df_train[df_train.columns] = scaler.transform(df_train[df_train.columns])
    df_val[df_val.columns] = scaler.transform(df_val[df_val.columns])
    df_test[df_test.columns] = scaler.transform(df_test[df_test.columns])

    # Save the training, validation and testing data
    print("Saving training, validation and testing data to csv files")
    df_train.to_csv(join(DATA_PATH, "train.csv"), index=False)
    df_val.to_csv(join(DATA_PATH, "val.csv"), index=False)
    df_test.to_csv(join(DATA_PATH, "test.csv"), index=False)

    return df_train, df_val, df_test


def make_data_subset(Ma: str = "0.84"):
    """From the original dataset, create a subset with a reduced number of snapshots, for only one Ma number and with interpolated data

    Args:
        Ma (str, optional): Ma number that should be included in the dataset. Defaults to "0.84".+
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
            cp[key] = interpolate_tensor(data_tensor=cp[key][:,:,:TARGET_SHAPE_TENSOR[2]])

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
    xx_new = interpolate(xx.unsqueeze(0).unsqueeze(0), size=TARGET_SHAPE_SLICE, mode="bilinear", align_corners=False).squeeze()
    yy_new = interpolate(yy.unsqueeze(0).unsqueeze(0), size=TARGET_SHAPE_SLICE, mode="bilinear", align_corners=False).squeeze()
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
    data_tensor_interp = pt.empty(TARGET_SHAPE_TENSOR)

    print("Interpolating ...")
    # loop over each timestep to interpolate the data
    for timestep in range(data_tensor.shape[2]):
        data_tensor_interp[:,:,timestep] = interpolate(data_tensor[:, :, timestep].unsqueeze(0).unsqueeze(0), size=TARGET_SHAPE_SLICE, mode="bilinear", align_corners=False).squeeze()

    print("Interpolated tensor shape:       ", data_tensor_interp.shape, "\n")
    return data_tensor_interp


def preprocessing():
    # load interpolated dataset
    data = pt.load(join(DATA_PATH, "cp_084_500snaps_interp.pt"))
    coords = pt.load(join(DATA_PATH, "coords_interp.pt"))
    grid_size = config.target_resolution[0] * config.target_resolution[1]
    xx, yy = coords
    print("Grid shape:                  ", xx.shape)

    x = xx.reshape([grid_size, 1]).squeeze()
    y = yy.reshape([grid_size, 1]).squeeze()
    print("Reshaped grid size:          ", x.shape)

    # identify keys for train data inside data dict
    train_keys = [key for key in list(data.keys()) if key not in config.nn_val_keys + config.nn_test_keys]
    train_data = data[train_keys[0]].flatten(0, 1)
    for train_key in train_keys[1:]:
        train_data = pt.concat((train_data, data[train_key].flatten(0, 1)), dim=1)
    print("Shape of train pressure data:        ", train_data.shape)
    train_tensor = reshape_data(x, y, train_data)

    val_data = data[config.nn_val_keys[0]].flatten(0, 1)
    for val_key in config.nn_val_keys[1:]:
        val_data = pt.concat((val_data, data[val_key].flatten(0, 1)), dim=1)
    print("Shape of validation pressure data:   ", val_data.shape)
    val_tensor = reshape_data(x, y, val_data)

    test_data = data[config.nn_test_keys[0]].flatten(0, 1)
    for test_key in config.nn_test_keys[1:]:
        test_data = pt.concat((test_data, data[test_key].flatten(0, 1)), dim=1)
    print("Shape of test pressure data:         ", test_data.shape)
    test_tensor = reshape_data(x, y, test_data)

    print("Fitting Scaler on training data")
    feature_scaler = StandardScaler().fit(train_tensor[:,1:])
    label_scaler = StandardScaler().fit(train_tensor[:,0])

    print("Making TensorDatasets with the scaled features and labels")
    train_dataset = TensorDataset(feature_scaler.scale(train_tensor[:,1:]), label_scaler.scale(train_tensor[:,0]).unsqueeze(-1))
    val_dataset = TensorDataset(feature_scaler.scale(val_tensor[:,1:]), label_scaler.scale(val_tensor[:,0]).unsqueeze(-1))
    test_dataset = TensorDataset(feature_scaler.scale(test_tensor[:,1:]), label_scaler.scale(test_tensor[:,0]).unsqueeze(-1))

    print("Saving ...")
    pt.save(train_dataset, join(DATA_PATH, "train_dataset.pt"))
    pt.save(val_dataset, join(DATA_PATH, "val_dataset.pt"))
    pt.save(test_dataset, join(DATA_PATH, "test_dataset.pt"))
    print("Done! \n")


def reshape_data(x, y, pressure_data):
    print("Reshaping ...")
    rows = pressure_data.shape[0] * pressure_data.shape[1]
    cols = 4
    pts_per_timestep = x.shape[0]
    time_steps = pressure_data.shape[1]
    
    pressure_data_resh = pressure_data.reshape([rows, 1]).squeeze()

    tensor = pt.zeros((rows, cols))
    for time_step in range(time_steps):
        start, end = time_step*pts_per_timestep, (time_step+1)*pts_per_timestep
        tensor[start:end, 1] = x
        tensor[start:end, 2] = y
        tensor[start:end, 3] = time_step%config.time_steps_per_cond
    tensor[:, 0] = pressure_data_resh

    print("Shape of data tensor:                ", tensor.shape, "\n")
    return tensor

if __name__ == "__main__":
    # preprocessing functions that interpolate coords from (465 x 159) to (256 x 128) and create a subset of the full dataset
    # interpolate_coords()
    # make_data_subset()
    preprocessing()