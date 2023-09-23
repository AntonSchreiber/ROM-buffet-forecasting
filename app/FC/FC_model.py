"""Simple LSTM network for time-series forecasting.

modified version of https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/building.html 
"""

import torch
from torch import nn

class FullyConnected(nn.Module):
    def __init__(self, latent_size: int, input_width: int, hidden_size: int, n_hidden_layers: int):
        """Create an FC instance.

        Args:
            latent_size (int): number of latent dimensions
            input_width (int): length of the input sequence
            hidden_size (int): number of neurons in each hidden layer
            n_hidden_layers (int): number of hidden layers
        """
        super().__init__()
        # the input layer is a concatenation of the input sequence
        self.input_size = input_width * latent_size
        self.output_size = latent_size
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers

        # input layer
        layer_list = [nn.Linear(self.input_size, self.hidden_size), nn.ReLU()]
        # hidden layers
        for _ in range(self.n_hidden_layers):
            layer_list.extend([nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU()])
        # output layer
        layer_list.append(nn.Linear(self.hidden_size, self.output_size))

        self.sequential = nn.Sequential(*layer_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x of shape: batch_size x (n_timesteps_in x n_latent)
        # output of shape batch_size x n_latent
        return self.sequential(x)

    def load(self, path: str="", device: str="cpu"):
        self.load_state_dict(torch.load(path, map_location=torch.device(device)))
    

# function to create a Fully-Connected network (not necessary, but for consistency)
def make_FC_model(latent_size: int, input_width: int, hidden_size: int, n_hidden_layers: int):
    return FullyConnected(
        latent_size=latent_size,
        input_width=input_width,
        hidden_size=hidden_size,
        n_hidden_layers=n_hidden_layers
    )