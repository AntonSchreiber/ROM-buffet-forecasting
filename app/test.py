import torch as pt
from time import sleep
from pathlib import Path
import os
from os.path import join

print(pt.__version__)

# use GPU if possible
device = pt.device("cuda") if pt.cuda.is_available() else pt.device("cpu")
print(device)

# Check if CUDA is available
cuda_available = pt.cuda.is_available()

if cuda_available:
    # Get the number of available CUDA devices
    num_cuda_devices = pt.cuda.device_count()

    if num_cuda_devices > 0:
        # Check if the first CUDA device (cuda:0) is available
        cuda_0_available = pt.cuda.device(0)
    else:
        cuda_0_available = False
else:
    cuda_0_available = False

print("CUDA Available:", cuda_available)
print("CUDA:0 Available:", cuda_0_available)

DATA_PATH = join(Path(os.path.abspath('')).parent, "data")
OUTPUT_PATH = join(Path(os.path.abspath('')).parent, "output", "VAE", "latent_study")

print("DATA_PATH:   ", Path(os.path.abspath('')))
print("OUTPUT_PATH: ", OUTPUT_PATH)