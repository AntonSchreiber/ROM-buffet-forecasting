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
    def __init__(self, latent_size: int, hidden_size: int, num_layers: int):
        """Create an LSTM instance.

        Args:
            latent_size (int): number of latent dimensions
            hidden_size (int): number of neurons in each NN of the LSTM
            num_layers (int): number of LSTMs stacked over each other
        """
        super().__init__()

        # assign inputs to attributes
        self.latent_size = latent_size 
        self.hidden_size = hidden_size   
        self.num_layers = num_layers          

        # define layers
        self.lstm = nn.LSTM(input_size=self.latent_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.latent_size)
        
    def forward(self, x: torch.Tensor, pred_horizon: int=1):
        # x: (batch_size, sequence_length, input_size)
        preds = []

        # propagate input through LSTM
        _, (h_n, c_n) = self.lstm(x)
        preds.append(self.fc(h_n[-1]))

        # if pred horizon > 1 continue to make predictions
        for _ in range(1, pred_horizon):
            x = self.shift_sequence(orig_seq=x, new_pred=preds[-1].unsqueeze(1))
            _, (h_n, c_n) = self.lstm(x, (h_n, c_n))
            preds.append(self.fc(h_n[-1]))

        return torch.stack(preds, dim=2).transpose(1, 2)
    
    def shift_sequence(self, orig_seq: torch.Tensor, new_pred: torch.Tensor):
        """
        Shift the sequence to remove the first timestep and append the new prediction to the sequence

        Parameters:
            orig_seq (torch.Tensor): The initial tensor of shape [batch_size, sequence_length, latent_size].
            new_pred (torch.Tensor): The new tensor with shape [batch_size, 1, latent_size] to be appended.

        Returns:
            torch.Tensor: The tensor with the first timestep removed and the new_pred appended.
                          The resulting shape will be [batch_size, sequence_length, latent_size].
        """    
        return torch.cat([orig_seq[:, 1:], new_pred], dim=1)
    
    def load(self, path: str="", device: str="cpu"):
        return self.load_state_dict(torch.load(path, map_location=torch.device(device)))