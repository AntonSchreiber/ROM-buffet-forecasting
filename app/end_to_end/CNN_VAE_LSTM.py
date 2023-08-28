import torch
from torch import nn
from CNN_VAE.CNN_VAE import ConvEncoder, ConvDecoder
from LSTM.LSTM_model import LSTM

class autoencoder_LSTM(nn.Module):
    """Wrapper class for convenient training.

    The implementation assumes that encoder, decoder and LSTM are compatible.
    """
    def __init__(self, encoder, LSTM, decoder) -> None:
        super(autoencoder_LSTM, self).__init__()
        self._encoder = encoder
        self._LSTM = LSTM
        self._decoder = decoder
    
    def forward(self, x, pred_horizon):
        # x is (batch_size, channels, height, width, sequence_length)
        # encoder takes (batch_size, channels, height, width) and outputs (batch_size, latent_size)
        x_enc = torch.stack([self._encoder(x[:, :, :, :, n]) for n in range(x.shape[4])], dim=2)

        # LSTM expects (batch_size, sequence_length, latent_size) and outputs (batch_size, pred_horizon, latent_size)
        x_pred = self._LSTM(x_enc.permute(0,2,1), pred_horizon)

        # decoder expects (batch_size, latent_size) and outputs (batch_size, channels, height, width)
        x_dec = torch.stack([self._decoder(x_pred[:, n]) for n in range(pred_horizon)], dim=4)
        return x_dec

    def save(self, path: str=""):
        torch.save(self._encoder.state_dict(), path + "_encoder.pt")
        torch.save(self._decoder.state_dict(), path + "_decoder.pt")
        torch.save(self._LSTM.state_dict(), path + "_LSTM.pt")

    def load(self, path: str="", device: str="cpu"):
        self._encoder.load_state_dict(torch.load(path + "_encoder.pt", map_location=torch.device(device)))
        self._decoder.load_state_dict(torch.load(path + "_decoder.pt", map_location=torch.device(device)))
        self._LSTM.load_state_dict(torch.load(path + "_LSTM.pt", map_location=torch.device(device)))