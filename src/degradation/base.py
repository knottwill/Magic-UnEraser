"""!@file base.py

@brief Contains base class for all degradation operators

@details Degradation operators are required in the training and sampling process of
a cold diffusion model, and it is important to know whether we are sampling or training.
The `sampling` method sets the sampling mode of the degradation operator. The `train` method
is a wrapper around the `train` method of the parent class (nn.Module), and sets the sampling mode to False.
"""
import torch.nn as nn


class DegredationOperator(nn.Module):
    """!
    @brief Base class for all degradation operators
    """

    def __init__(self):
        super().__init__()
        self.sampling_mode = False

    def sampling(self, mode: bool = True):
        """!
        @brief Set the sampling mode of the degradation operator
        """
        self.sampling_mode = mode

    def train(self, mode: bool = True):
        """!
        @brief Set the sampling mode to False if we are in training mode
        """
        super().train(mode)
        if mode:  # if we are in training mode, we are not sampling
            self.sampling_mode = False
