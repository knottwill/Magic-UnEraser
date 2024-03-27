"""!@file cnn.py

@brief We create a simple 2D convolutional neural network to use as the reconstructor
network within the diffusion models.

@details The CNN is a stack of convolutional blocks, each of which consists of a
convolutional layer, followed by a layer normalization and a GELU activation.


First, we create a single CNN block which we will stack to create the
full network. We use `LayerNorm` for stable training and no batch dependence.

We then create the full CNN model, which is a stack of these blocks
according to the `n_hidden` tuple, which specifies the number of
channels at each hidden layer.
"""

import torch
import torch.nn as nn
import numpy as np


class CNNBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        expected_shape,
        act=nn.GELU,
        kernel_size=7,
    ):
        super().__init__()

        self.net = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2), nn.LayerNorm((out_channels, *expected_shape)), act())

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(
        self,
        in_channels,
        expected_shape=(28, 28),
        n_hidden=(64, 128, 64),
        kernel_size=7,
        last_kernel_size=3,
        time_embeddings=16,
        act=nn.GELU,
    ) -> None:
        super().__init__()
        last = in_channels

        self.blocks = nn.ModuleList()
        for hidden in n_hidden:
            self.blocks.append(
                CNNBlock(
                    last,
                    hidden,
                    expected_shape=expected_shape,
                    kernel_size=kernel_size,
                    act=act,
                )
            )
            last = hidden

        # The final layer, we use a regular Conv2d to get the
        # correct scale and shape (and avoid applying the activation)
        self.blocks.append(
            nn.Conv2d(
                last,
                in_channels,
                last_kernel_size,
                padding=last_kernel_size // 2,
            )
        )

        # This part is literally just to put the single scalar "s = t/T" into the CNN
        # in a nice, high-dimensional way:
        self.time_embed = nn.Sequential(
            nn.Linear(time_embeddings * 2, 128),
            act(),
            nn.Linear(128, 128),
            act(),
            nn.Linear(128, 128),
            act(),
            nn.Linear(128, n_hidden[0]),
        )
        frequencies = torch.tensor([0] + [2 * np.pi * 1.5**i for i in range(time_embeddings - 1)])
        self.register_buffer("frequencies", frequencies)

    def time_encoding(self, s: torch.Tensor) -> torch.Tensor:
        """! @brief Encode the time step into the latent space

        @param s: Severity (timestep/T) -  a scalar in [0, 1]
        """

        phases = torch.concat(
            (
                torch.sin(s[:, None] * self.frequencies[None, :]),
                torch.cos(s[:, None] * self.frequencies[None, :]) - 1,
            ),
            dim=1,
        )

        return self.time_embed(phases)[:, :, None, None]

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """! @brief Forward pass through the CNN

        @param x: Input tensor
        @param s: Severity (timestep/T) - a scalar in [0, 1]
        """

        # Initial embedding and time encoding
        embed = self.blocks[0](x)
        embed += self.time_encoding(s)

        # Forward pass through the rest of the blocks
        for block in self.blocks[1:]:
            embed = block(embed)

        return embed
