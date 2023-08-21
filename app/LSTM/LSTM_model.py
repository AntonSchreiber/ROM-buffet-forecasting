"""Simple LSTM network for time-series forecasting.

Some helpful links used as reference for the implementation:
- https://www.youtube.com/watch?v=Cdmq0xtFG5w&t=1s
- https://towardsdatascience.com/pytorch-lstms-for-time-series-data-cd16190929d7 
- https://cnvrg.io/pytorch-lstm/ 
- https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

"""

import torch
import torch.nn as nn
from utils.helper_funcs import shift_input_sequence

class LSTM(nn.Module):
    def __init__(self, latent_size, hidden_size, num_layers):
        super().__init__()

        # assign inputs to attributes
        self.latent_size = latent_size              # input and output size of the network
        self.hidden_size = hidden_size              # number of neurons in each NN of the LSTM
        self.num_layers = num_layers                # number of LSTMs stacked over each other

        # define layers
        self.lstm = nn.LSTM(input_size=self.latent_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.latent_size)

    def init(self, x: torch.Tensor, device: str):
        batch_size = x.shape[0]

        # initialize hidden state and cell state
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float32).to(device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float32).to(device)

        return h_0, c_0
        
    def forward(self, x: torch.Tensor, device: str="cpu"):
        # x: (batch_size, sequence_length, input_size)

        # propagate input through LSTM
        output, (h_n, _) = self.lstm(x, self.init(x, device))
        output = self.fc(h_n[-1])

        return output
    
    def load(self, path: str="", device: str="cpu"):
        self.load_state_dict(torch.load(path, map_location=torch.device(device)))


if __name__ == "__main__":
    hidden_size = 32
    num_layers = 2

    batch_size = 256
    latent_size = 32
    input_width = 16

    # (batch_size x latent_size, input_width)
    test_input = torch.rand(([batch_size, input_width, latent_size]))   

    # Instantiate the LSTM model
    lstm_model = LSTM(latent_size=latent_size, hidden_size=hidden_size, num_layers=num_layers)

    # Forward pass
    output = lstm_model(test_input)

    print("Input Shape:", test_input.shape)
    print("Output Shape:", output.shape)