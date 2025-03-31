# from models import EDSR_2D
# Import other models as needed

def get_model(model_name, config):
    """
    Factory function to create model instances
    
    Args:
        model_name: String identifier for the model
        config: Configuration dictionary with model parameters
    
    Returns:
        Instantiated model
    """
    if model_name == 'EDSR_2D':
        from models.EDSR_2D import EDSR
        return EDSR(
            num_channels=config['num_channels'],
            n_resblock=config['n_resblock'],
            filters=config['filters'],
            res_scaling=config['res_scaling'],
            scale=config['scale']
        )
    # Add other models as needed
    else:
        raise ValueError(f"Unknown model: {model_name}")