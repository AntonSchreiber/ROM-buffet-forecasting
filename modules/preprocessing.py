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


def load_data(filename: str) -> pt.Tensor:
    """Load a file from filename in data folder

    Args:
        filename (string): path to file relative to data folder

    Returns:
        pt.Tensor: Data Tensor
    """
    data = pt.load(join(data_path, filename))
    return data


def reshape_data(x: pt.Tensor, y: pt.Tensor, cp: dict, cases: dict) -> pt.Tensor:  
    """Reshapes all of the data into the desired shape for the models

    Args:
        x (pt.Tensor): x-coordinates array
        y (pt.Tensor): y-coordinates array
        cp (dict): time dependant pressure coefficient data
        cases (dict): dict with the case names as keys and the filtered Ma and alpha as vals

    Returns:
        pt.Tensor: Fully assembled data tensor
    """
    # Extract information about the data dimesnions
    dataset_len = sum([val.numel() for key, val in cp.items()])
    keys = list(cp.keys())
    vals_per_step = cp[keys[0]].shape[0]*cp[keys[0]].shape[1]
    num_timesteps = cp[keys[0]].shape[2]

    # clone x and y by the amount of timesteps to achieve desired dimension
    x = pt.tile(x, (num_timesteps,))
    y = pt.tile(y, (num_timesteps,))

    # creating array with timesteps 
    t = pt.zeros((cp[keys[0]].numel(),))
    for step in range(1, num_timesteps):
        t1, t2= step*vals_per_step, (step+1)*vals_per_step
        t[t1:t2] = step

    data = pt.zeros((dataset_len, 6))
    # Loop over Ma-alpha confs
    for i, (key, vals) in enumerate(cp.items()):
        # start and end indices for current conf in data tensor
        start, end = i*vals.numel(), (i+1)*vals.numel()
   
        # assigning data to corresponding tensor column
        data[start:end, 0] = x
        data[start:end, 1] = y
        data[start:end, 2] = pt.full((vals.numel(),), cases[key][0])
        data[start:end, 3] = pt.full((vals.numel(),), cases[key][0])
        data[start:end, 4] = t
        data[start:end, 5] = pt.reshape(vals, (vals.numel(), 1)).squeeze()

    print("Dataset size     =", data.shape, "\n")

    return data


def get_coords() -> tuple[pt.Tensor, pt.Tensor]:
    """Read in the x and y coordinates from the coords.pt file and convert the grid to arrays

    Returns:
        tuple[pt.Tensor, pt.Tensor]: x and y coordinates arrays
    """
    # Load the coordinate grid
    coords = load_data("coords.pt")
    x_grid, y_grid = coords["ma0.84_alpha4.00"]

    # reshape 2D grid to 1D array
    x= pt.reshape(x_grid, (x_grid.numel(), 1)).squeeze()
    y= pt.reshape(y_grid, (y_grid.numel(), 1)).squeeze()
    
    return x, y

def get_cp_and_cases() -> tuple[pt.Tensor, dict]:
    """Load the raw cp data and extract the Ma and alpha

    Returns:
        tuple[pt.Tensor, dict]: cp tensor and dict with the case names as keys and the filtered Ma and alpha as vals
    """
    # Load the cp data
    cp = load_data("cp_test.pt")

    # Get the conf names
    keys = list(cp.keys())

    # Delete the "ma" and "_alpha" from the keys and convert to float
    cases = [key.replace("ma", "").replace("_alpha", "") for key in list(cp.keys())]
    cases_dict = {keys[i]: [float(case[:4]), float(case[4:])] for i, case in enumerate(cases)}

    return cp, cases_dict


if __name__ == "__main__":
    # Out-of-Loop Pre-Processing to assemble the global dataset
    print("Loading data")
    x, y = get_coords()
    cp, cases = get_cp_and_cases()
    
    # Reshape all of the data into one tensor
    print("Reshaping data")
    data = reshape_data(x, y, cp, cases)
    print("Data reshaped")

    # Save the data as a TensorDataset
    print("Creating TensorDataset")
    dataset = pt.utils.data.TensorDataset(data[:, :5], data[:, 5])
    pt.save(dataset, join(data_path, "dataset.pt"))
    print("Dataset saved")