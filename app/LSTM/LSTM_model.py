"""Simple LSTM network for time-series forecasting.

Some helpful links used as reference for the implementation:
- https://www.youtube.com/watch?v=Cdmq0xtFG5w&t=1s
- https://towardsdatascience.com/pytorch-lstms-for-time-series-data-cd16190929d7 
- https://cnvrg.io/pytorch-lstm/ 
- https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

"""

import torch
import torch.nn as nn

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
        
    def forward(self, x: torch.Tensor, device: str="cpu", pred_horizon: int=1):
        # x: (batch_size, sequence_length, input_size)
        preds = []

        # propagate input through LSTM
        _, (h_n, c_n) = self.lstm(x, self.init(x, device))
        preds.append(self.fc(h_n[-1]))

        # if pred horizon > 1 continue o make predictions
        for _ in range(1, pred_horizon):
            x = self.shift_sequence(orig_seq=x, new_pred=preds[-1].unsqueeze(1))
            _, (h_n, c_n) = self.lstm(x, (h_n, c_n))
            preds.append(self.fc(h_n[-1]))

        return torch.stack(preds, dim=2).transpose(1, 2)
    
    def shift_sequence(self, orig_seq, new_pred):
        """
        Shift the tensor to remove the first timestep and append the new prediction to the sequence

        Parameters:
            orig_seq (torch.Tensor): The initial tensor of shape [batch_size, sequence_length, latent_size].
            new_pred (torch.Tensor): The new tensor with shape [batch_size, 1, latent_size] to be appended.

        Returns:
            torch.Tensor: The tensor with the first timestep removed and the new_pred appended.
                          The resulting shape will be [batch_size, sequence_length, latent_size].
        """    
        return torch.cat([orig_seq[:, 1:, :], new_pred], dim=1)
    
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

    print("Input Shape:", test_input.shape)         # batch_size, input_width, latent_size
    print("Output Shape:", output.shape)            # batch_size, pred_horizon, latent size