"""!
@brief Contains base class for all degradation operators
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
        super().train(mode)
        if mode: # if we are in training mode, we are not sampling
            self.sampling_mode = False
