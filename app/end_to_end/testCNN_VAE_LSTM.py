import unittest
import torch
from CNN_VAE.CNN_VAE import ConvEncoder, ConvDecoder
from LSTM.LSTM_model import LSTM
from CNN_VAE_LSTM import autoencoder_LSTM  
from utils import config

class TestAutoencoderLSTM(unittest.TestCase):
    def setUp(self):
        self.input_width = 32   # Example input width
        latent_size = 64        # Example latent size

        self.encoder = ConvEncoder(
        in_size=config.target_resolution,
        n_channels=config.VAE_input_channels,
        n_latent=latent_size,
        variational=True,
        layernorm=True
        )

        self.lstm = LSTM(
            latent_size=latent_size, 
            hidden_size=2, 
            num_layers=2)

        self.decoder = ConvDecoder(
            in_size=config.target_resolution,
            n_channels=config.VAE_output_channels,
            n_latent=latent_size,
            layernorm=True,
            squash_output=True
        )

        self.model = autoencoder_LSTM(self.encoder, self.lstm, self.decoder)

    def test_shapes(self):
        batch_size = 64
        channels = 1
        height = 256
        width = 128
        input_width = 10
        pred_horizon = 1

        input_data = torch.randn(batch_size, channels, height, width, input_width)

        output = self.model(input_data, pred_horizon=pred_horizon)

        expected_output_shape = (batch_size, channels, height, width, pred_horizon)
        self.assertEqual(output.shape, expected_output_shape)

if __name__ == '__main__':
    unittest.main()