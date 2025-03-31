"""Default configuration parameters"""

default_config = {
    # Dataset parameters
    'dataset': 'DIV2K_vol_2D',
    'data_root': r"C:\Users\Patrick\Downloads\EDSR_Battery\data",
    # 'split': 'train',
    # 'scale': 4,
    'file_extension': 'tiff',
    'volume_depth': 330,
    # 'max_samples': None,
    # 'patch_size': 64,
    # 'patch_stride': 48,

    'scale': 2,
    'batch_size': 1,

    # Model parameters
    'model': 'EDSR_2D',
    # 'num_features': 64,
    # 'num_blocks': 16,
    'num_channels': 3,
    'filters' : 256,
    'n_resblock': 32,
    'res_scaling': 0.1,

    # Training parameters
    'num_epochs': 100,
    'learning_rate': 0.0001,
    'scheduler_patience': 10,
    'scheduler_factor': 0.5,
    'loss_function': 'L1',  # Options: 'L1', 'MSE', 'SmoothL1',
    'optimizer': 'Adam',  # Options: 'Adam', 'SGD', 'RMSprop'
    'momentum': 0.9,  # Only used for SGD

    # System parameters
    'num_workers': 0, # for Windows, set to 0
    'pin_memory': True, # for windows, set to False
    'device': 'cuda',
    'save_dir': './ckpt',
    'checkpoint_frequency': 5
}