"""Simple convolutional autoencoder for building ROMs.

Some helpful links used as reference for the implementation:
- https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html
- https://avandekleut.github.io/vae/
- https://debuggercafe.com/convolutional-variational-autoencoder-in-pytorch-on-mnist-dataset/
- https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians

"""

from typing import Callable, Tuple
from math import prod, log2
import torch
from torch import nn
import torch.nn.functional as F


def power_of_two(shape: Tuple[int]) -> bool:
    """Test if all numbers in a tuple are powers of two.

    Reference for the implementation:
    https://stackoverflow.com/questions/57025836/how-to-check-if-a-given-number-is-a-power-of-twonce:


    :param shape: numbers to test
    :type shape: Tuple[int]
    :return: True if all elements are powers of two
    :rtype: bool
    """
    p2 = [log2(i).is_integer() for i in shape]
    return all(p2)


class ConvEncoder(nn.Module):
    def __init__(self,
                 in_size: Tuple[int],
                 n_channels: Tuple[int],
                 n_latent: int,
                 activation: Callable=F.relu,
                 batchnorm: bool = False,
                 layernorm: bool = False,
                 variational: bool = False
                 ):
        """Create a convolutional encoder instance.

        The implementation assumes that the number of elements in each
        direction of the input data is devisable by two.

        :param in_size: size of the input data
        :type in_size: Tuple[int]
        :param n_channels: number of channels; the first entry corresponds to
          the input's number of channels (typically 1).
        :type n_channels: Tuple[int]
        :param n_latent: number of latent/bottleneck neurons
        :type n_latent: int
        :param activation: activation function, defaults to F.relu
        :type activation: Callable, optional
        :param batchnorm: batch normalization is applied to each convolutional
          layer if True, defaults to False
        :type batchnorm: bool, optional
        :param variational: latent features are sampled from a normal distribution
          if True and KL-divergence is computed to train the encoder, defaults to False
        :type variational: bool, optional
        """
        super(ConvEncoder, self).__init__()
        self._in_size = in_size
        self._n_channels = n_channels
        self._n_latent = n_latent
        self._activation = activation
        self._batchnorm = batchnorm
        self._layernorm = layernorm
        self._variational = variational

        # check input dimensions
        assert power_of_two(self._in_size), "Input sizes must be a power of two."

        # create convolutional layers with optional batch normalization
        self._layers = nn.ModuleList()
        n_conv_layers = len(self._n_channels) - 1
        for i in range(n_conv_layers):
            self._layers.append(
                nn.Conv2d(
                    self._n_channels[i], self._n_channels[i+1], kernel_size=2, stride=2)
            )
            if self._batchnorm:
                self._layers.append(nn.BatchNorm2d(self._n_channels[i+1]))
            elif self._layernorm:
                current_size = [s//2**(i+1) for s in in_size]
                self._layers.append(nn.LayerNorm([self._n_channels[i+1], current_size[0], current_size[1]]))

        # add fully-connected layer after last convolution
        # these formulas only hold if kernel size and stride are kept at two
        last_size = [s//2**n_conv_layers for s in in_size]
        features_flat = self._n_channels[-1] * prod(last_size)
        self._latent_mean = nn.Linear(features_flat, self._n_latent)
        if self._variational:
            self._latent_log_var = nn.Linear(features_flat, self._n_latent)
            self._dist = torch.distributions.Normal(0, 1)
            self.divergence = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass through encoder.

        :param x: input data of dimension BxCxHxW (batch size, channels
          height, and width)
        :type x: pt.Tensor
        :return: latent features of dimension BxL (batch size, latent features)
        :rtype: pt.Tensor
        """
        for layer in self._layers:
            x = self._activation(layer(x))
        x = torch.flatten(x, start_dim=1)
        if self._variational:
            mean = self._latent_mean(x)
            log_var = self._latent_log_var(x)
            self.divergence = -0.5 * (1 + log_var - mean**2 - log_var.exp()).sum()
            return mean + self._dist.sample(mean.shape).to(x.device) * torch.exp(0.5*log_var)
        else:
            return self._latent_mean(x)


class ConvDecoder(nn.Module):
    def __init__(self,
                 in_size: Tuple[int],
                 n_channels: Tuple[int],
                 n_latent: int,
                 activation: Callable=F.relu,
                 batchnorm: bool = False,
                 layernorm: bool = False,
                 squash_output: bool = False
                 ):
        """Create a convolutional decoder instance.

        The implementation assumes that the number of elements in each
        direction of the input data is devisable by two.

        :param in_size: size of the encoder's input data
        :type in_size: Tuple[int]
        :param n_channels: number of channels; the last entry corresponds to
          the number of channels of the original encoder input (typically 1).
        :type n_channels: Tuple[int]
        :param n_latent: number of latent/bottleneck neurons
        :type n_latent: int
        :param activation: activation function, defaults to F.relu
        :type activation: Callable, optional
        :param batchnorm: batch normalization is applied after each
          convolutional layer if True, defaults to False
        :type batchnorm: bool, optional
        :param squash_output: output is limited to the range [-1, 1] by using
          a tanh activation function, defaults to False
        :type squash_output: bool, optional
        """
        super(ConvDecoder, self).__init__()
        self._in_size = in_size
        self._n_channels = n_channels
        self._n_latent = n_latent
        self._activation = activation
        self._batchnorm = batchnorm
        self._layernorm = layernorm
        self._squash_output = squash_output

        # create fully-connected layer as adapter between latent
        # variables and first convolution
        self._layers = nn.ModuleList()
        n_conv_layers = len(self._n_channels) - 1
        self._first_size = [s//2**n_conv_layers for s in in_size]
        features_flat = self._n_channels[0] * prod(self._first_size)
        self._latent = nn.Linear(self._n_latent, features_flat)

        # add convolutional layers with optional batch normalization
        for i in range(n_conv_layers):
            self._layers.append(
                nn.ConvTranspose2d(
                    self._n_channels[i], self._n_channels[i+1], kernel_size=2, stride=2)
            )
            if self._batchnorm:
                self._layers.append(nn.BatchNorm2d(self._n_channels[i+1]))
            elif self._layernorm:
                current_size = [s * 2**(i+1) for s in self._first_size]
                self._layers.append(nn.LayerNorm([self._n_channels[i+1], current_size[0], current_size[1]]))
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass through decoder.

        :param x: latent features of dimension BxL (batch size, latent features)
        :type x: pt.Tensor
        :return: original input data of dimension BxCxHxW (batch size, channels
          height, and width)
        :rtype: pt.Tensor
        """
        x = self._latent(x)
        x = x.reshape(x.shape[0], self._n_channels[0], *self._first_size)
        for layer in self._layers[:-1]:
            x = self._activation(layer(x))
        if self._squash_output:
            return torch.tanh(self._layers[-1](x))
        else:
            return self._layers[-1](x)


class Autoencoder(nn.Module):
    """Wrapper class for convenient training.

    The implementation assumes that encoder and decoder
    are compatible.
    """
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
        self._encoder = encoder
        self._decoder = decoder

    def forward(self, x):
        return self._decoder(self._encoder(x))

    def save(self, path: str=""):
        print(path + "_encoder.pt")
        torch.save(self._encoder.state_dict(), path + "_encoder.pt")
        torch.save(self._decoder.state_dict(), path + "_decoder.pt")

    def load(self, path: str=""):
        self._encoder.load_state_dict(torch.load(path + "_encoder.pt"))
        self._decoder.load_state_dict(torch.load(path + "_decoder.pt"))