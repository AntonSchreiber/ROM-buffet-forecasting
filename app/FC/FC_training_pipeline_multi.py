import sys
import os
from os.path import join
from pathlib import Path
from itertools import product
from collections import defaultdict
import torch as pt
import torch.nn as nn
from torch.utils.data import DataLoader
pt.manual_seed(0)

# include app directory into sys.path
REMOTE= True
parent_dir = Path(os.path.abspath('')).parent.parent if REMOTE else Path(os.path.abspath(''))
app_dir = join(parent_dir, "app")
if app_dir not in sys.path:
      sys.path.append(app_dir)

from utils.DataWindow import DataWindow
from FC.FullyConnected import FullyConnected
from utils.EarlyStopper import EarlyStopper
from utils.training_funcs import train_AR_pred
from utils.helper_funcs import delete_directory_contents, reduce_datasets_SVD_multi, reduce_datasets_VAE_multi
import utils.config as config

# use GPU if possible
device = pt.device("cuda") if pt.cuda.is_available() else pt.device("cpu")
print("Computing device:        ", device)

# define prediction horizon and type of dimensionality reduction
PRED_HORIZON = 32
DIM_REDUCTION = "VAE"       # one of ("SVD" / "VAE")
N_LATENT = config.SVD_rank if DIM_REDUCTION == "SVD" else config.VAE_latent_size
BATCH_SIZE = config.FC_SVD_multi_batch_size if DIM_REDUCTION == "SVD" else config.FC_VAE_multi_batch_size

# define paths
VAE_PATH = join(parent_dir, "output", "VAE", "latent_study", config.VAE_model)
SVD_PATH = join(parent_dir, "output", "SVD", "U.pt")
DATA_PATH = join(parent_dir, "data", "multi_flow_cond")
OUTPUT_PATH = join(parent_dir, "output", "FC", "multi", DIM_REDUCTION, "param_study", f"pred_horizon_{PRED_HORIZON}")

# define study parameters of Fully-Connected network
INPUT_WIDTHS = [32]
HIDDEN_SIZES = [32, 64, 128, 256]
N_HIDDEN_LAYERS = [1, 2]

def start_study(n_repeat):
    print("Training Fully-Connected models with varying model parameters: ")
    print("     input width:                ", INPUT_WIDTHS)
    print("     neurons in hidden layers:   ", HIDDEN_SIZES)
    print("     number of hidden layers:    ", N_HIDDEN_LAYERS)

    delete_directory_contents(OUTPUT_PATH)

    # compress dataset into reduced state either by VAE or SVD
    if DIM_REDUCTION == "VAE":
        (train_red, val_red, test_red), _ = reduce_datasets_VAE_multi(DATA_PATH, VAE_PATH, OUTPUT_PATH, device) 
    elif DIM_REDUCTION == "SVD":
        (train_red, val_red, test_red), _ = reduce_datasets_SVD_multi(DATA_PATH, SVD_PATH, OUTPUT_PATH) 
    else:
        raise ValueError("Unknown DIM_REDUCTION")

    # start study
    print("Starting study...")
    study_results = defaultdict(list)
    param_combinations = list(product(INPUT_WIDTHS, HIDDEN_SIZES, N_HIDDEN_LAYERS))
    print(f"'---> {len(param_combinations) * n_repeat} trainings in total")

    for i in range(n_repeat):
        print("---Iteration %d--------" %(i+1))
        pt.manual_seed(i)
        for param_set in param_combinations:
            input_width, hidden_size, n_hidden_layers = param_set
            set_key = f"{input_width}_{hidden_size}_{n_hidden_layers}"
            print("--input_width={}, hidden_size={} and n_hidden={}".format(input_width, hidden_size, n_hidden_layers))

            # create DataWindow object to create windows of data, feed into DataLoaders
            data_window = DataWindow(train=train_red, val=val_red, test=test_red, input_width=input_width, pred_horizon=PRED_HORIZON)

            train_loader = DataLoader(data_window.train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(data_window.val_dataset, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(data_window.test_dataset, batch_size=BATCH_SIZE, shuffle=True)
            
            # initialize model and utilities
            model = FullyConnected(
                latent_size=N_LATENT,
                input_width=input_width,
                hidden_size=hidden_size,
                n_hidden_layers=n_hidden_layers
        )

            loss_func_latent = nn.MSELoss()
            optimizer = pt.optim.Adam(model.parameters(), lr=1e-3)
            scheduler = pt.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", patience=config.FC_patience_scheduler, factor=config.FC_lr_factor)

            # start training and append resoults to defaultdict
            study_results[f"{input_width}_{hidden_size}_{n_hidden_layers}"].append(train_AR_pred(
                model=model,
                loss_func=loss_func_latent,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                optimizer=optimizer,
                lr_schedule=scheduler,
                epochs=config.FC_multi_epochs,
                device=device
            ))
            pt.save(model.state_dict(), join(OUTPUT_PATH, str(i + 1) + "_" + set_key + ".pt"))
            print("\n")
        
    # save results of training metrics
    print("========== Study finished, saving results")
    pt.save(study_results, join(OUTPUT_PATH, "study_results.pt"))


if __name__ == '__main__':
    start_study(n_repeat=1)

    
