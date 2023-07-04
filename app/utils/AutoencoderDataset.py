import torch as pt
from torch.utils.data import Dataset


class AutoencoderDataset(Dataset):
    def __init__(self, cp: pt.Tensor, yy: pt.Tensor, xx: pt.Tensor) -> None:
        """Custom dataset class for surface pressure datasets to train a convolutional autoencoder

        Args:
            pressure_data (pt.Tensor): surface pressure data ("images")
            yy (pt.Tensor): y-coordinates of pressure data of shape (height, width)
            xx (pt.Tensor): x-coordinates of pressure data of shape (height, width)
        """
        self.cp = cp
        self.yy = yy
        self.xx = xx

    def __getitem__(self, index: int) -> tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        """Retrieves the surface pressure image, x mesh and y mesh at the given index

        Args:
            index (int): The index of the sample to retrieve

        Returns:
            tuple[pt.Tensor, pt.Tensor, pt.Tensor]: A tuple containing the surface pressure image, x mesh and y mesh as torch tensors
        """
        surface_pressure_image = self.cp[:, :, index].unsqueeze(0) # Add a channel dimension
        yy = self.yy
        xx = self.xx
        return surface_pressure_image, yy, xx
    def __len__(self) -> int:
        """Returns the total number of samples/ timesteps in the dataset

        Returns:
            int: The number of samples or timesteps in the dataset
        """
        return self.cp.shape[2]