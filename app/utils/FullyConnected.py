# class for a simple fully-connected network architecture as baseline model of this study
# based on https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/building.html 
import torch
from torch import nn

class FullyConnected(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, n_hidden_layers: int):
        super().__init__()
        assert input_size % output_size == 0 or output_size % input_size == 0, f"Network input size {input_size} and output size {output_size} must be dividable without remainder"
        self.input_size = input_size
        self.output_size = output_size

        # input layer
        module_list = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        # hidden layers
        for _ in range(n_hidden_layers):
            module_list.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        # output layer
        module_list.append(nn.Linear(hidden_size, output_size))

        self.sequential = nn.Sequential(*module_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x of shape: batch_size x (n_timesteps_in x n_latent)
        # output of shape batch_size x (n_timesteps_out x n_latent)
        return self.sequential(x)
    

if __name__ == '__main__':
    fc = FullyConnected(
        input_size=8,
        output_size=1,
        hidden_size=5,
        n_hidden_layers=2)
    print(fc)

    data = torch.rand(32, 8)
    print("Input shape:     ", data.shape)
    data = fc(data)
    print("Output shape:    ", data.shape)