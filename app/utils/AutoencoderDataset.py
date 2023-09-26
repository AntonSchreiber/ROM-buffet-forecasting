import torch as pt
from torch.utils.data import Dataset

class AutoencoderDataset(Dataset):
    def __init__(self, cp: pt.Tensor) -> None:
        """Custom class for surface pressure datasets to train a convolutional autoencoder neural network

        Args:
            pressure_data (pt.Tensor): surface pressure snapshots [height, width, n_snapshots]
        """
        self.cp = cp

    def __getitem__(self, index: int) -> tuple:
        """Retrieves the surface pressure image at the given index

        Args:
            index (int): The index of the sample to retrieve

        Returns:
            pt.Tensor: A surface pressure image with [1, height, width]
        """
        return self.cp[:, :, index].unsqueeze(0) # Add a channel dimension
    
    def __len__(self) -> int:
        """Returns the total number of samples/ timesteps in the dataset

        Returns:
            int: The number of samples or timesteps in the dataset
        """
        return self.cp.shape[2]