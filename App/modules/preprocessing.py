from os.path import join
import torch as pt
from torch.nn.functional import interpolate
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

DATA_PATH = "./data"
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


def reshape_data(x: pt.Tensor, y: pt.Tensor, cp: dict, cases: dict) -> pt.Tensor:  
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


def to_dataframe(data: np.ndarray) -> pd.DataFrame:
    """Convert a numpy array to a pandas dataframe

    Args:
        data (np.ndarray): Data array

    Returns:
        pd.DataFrame: Dataframe
    """
    column_names = ["x", "y", "Ma", "alpha", "t", "cp"]
    df = pd.DataFrame(data, columns=column_names)

    print(df.info())
    print(df.sample(5))

    print("Data converted to pandas dataframe")

    return df


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
        Ma (str, optional): Ma number that should be included in the dataset. Defaults to "0.84".
        num_snapshots (int, optional): Number of snapshots that should be included in the dataset. Defaults to 500.
    """
    # load dataset and extract keys
    cp = load_data("cp_clean.pt")
    keys = list(cp.keys())

    # get the original and new coordinate tuples
    coords_orig, coords_new = get_coord_arrays()

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
    pt.save(cp, "./data/cp_084_500snaps_interp.pt")
    print("Done!")


def get_coord_arrays() -> tuple[tuple, tuple]:
    """Define x and y coordinates in the format required for RegularGridInterpolator

    Returns:
        tuple[tuple, tuple]: tuple of tuples with x and y coordinates in array format
    """
    orig_res = (465, 159)
    x_orig = np.linspace(0, orig_res[0] - 1, orig_res[0])
    y_orig = np.linspace(0, orig_res[1] - 1, orig_res[1])

    new_res = (256, 128)
    x_new = np.linspace(0, orig_res[0] - 1, new_res[0])
    y_new = np.linspace(0, orig_res[1] - 1, new_res[1])

    return (x_orig, y_orig), (x_new, y_new)


def plot_interpolation(coords_orig, coords_new, tensor, interpolated_tensor):
    x_mesh_orig, y_mesh_orig = pt.meshgrid(coords_orig[0], coords_orig[1], indexing='ij')
    x_mesh_new, y_mesh_new = pt.meshgrid(coords_new[0], coords_new[1], indexing='ij')

    mean, std = tensor[:, :, 0].mean(), tensor[:, :, 0].std()
    vmin, vmax = mean - 2*std, mean + 2*std
    levels = pt.linspace(vmin, vmax, 120)
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    cont1 = ax1.contourf(x_mesh_orig, y_mesh_orig, tensor[:, :, 0], vmin=vmin, vmax=vmax, levels=levels, extend="both")
    cont2 = ax2.contourf(x_mesh_new, y_mesh_new, interpolated_tensor[:, :, 0], vmin=vmin, vmax=vmax, levels=levels, extend="both")


def create_new_coords(orig_res: tuple = (465, 159), new_res: tuple = (256, 128)):
    """Create new coordinate meshes for interpolated data

    Args:
        orig_res (tuple, optional): original data resolution. Defaults to (465, 159).
        new_res (tuple, optional): new data resolution. Defaults to (256, 128).
    """
    # create x and y arrays and convert them to meshes
    x_new = pt.linspace(0, orig_res[0] - 1, new_res[0])
    y_new = pt.linspace(0, orig_res[1] - 1, new_res[1])
    x_mesh, y_mesh = pt.meshgrid(x_new, y_new, indexing='ij')

    # save as .pt
    pt.save((x_mesh, y_mesh), "./data/coords_interp.pt")


def interpolate_coords():
    print("INTERPOLATING COORDINATES")

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


def interpolate_tensor(data_tensor: pt.Tensor):
    print("INTERPOLATING DATA TENSOR")
    print("Original tensor shape:           ", data_tensor.shape)

    data_tensor_interp = pt.empty(TARGET_SHAPE_TENSOR)
    print("Interpolating ...")
    for timestep in range(data_tensor.shape[2]):
        data_tensor_interp[:,:,timestep] = interpolate(data_tensor[:, :, timestep].unsqueeze(0).unsqueeze(0), size=TARGET_SHAPE_SLICE, mode="bilinear", align_corners=False).squeeze()

    print("Interpolated tensor shape:       ", data_tensor_interp.shape, "\n")
    return(data_tensor)

if __name__ == "__main__":
    # interpolate_coords()
    make_data_subset()