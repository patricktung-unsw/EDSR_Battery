import torch.nn as nn

def get_loss_function(loss_name):
    """
    Factory function to create loss functions.

    Args:
        loss_name: String identifier for the loss function (e.g., 'L1', 'MSE').

    Returns:
        Instantiated loss function.
    """
    if loss_name == 'L1':
        return nn.L1Loss()
    elif loss_name == 'MSE':
        return nn.MSELoss()
    elif loss_name == 'SmoothL1':
        return nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")