import torch

def get_optimizer(optimizer_name, model, config):
    """
    Factory function to create optimizers.

    Args:
        optimizer_name: String identifier for the optimizer (e.g., 'Adam', 'SGD').
        model: The model whose parameters will be optimized.
        config: Configuration dictionary with optimizer parameters.

    Returns:
        Instantiated optimizer.
    """
    if optimizer_name == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif optimizer_name == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr=config['learning_rate'],
            momentum=config.get('momentum', 0.9)
        )
    elif optimizer_name == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr=config['learning_rate'],
            alpha=0.99
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")