from datasets.DIV2K_vol_2D import DIV2K

def get_dataset(dataset_name, split, config):
    """
    Factory function to create dataset instances.

    Args:
        dataset_name: String identifier for the dataset (e.g., 'DIV2K').
        split: 'train' or 'valid' to specify the dataset split.
        config: Configuration dictionary with dataset parameters.

    Returns:
        Instantiated dataset object.
    """
    if dataset_name == 'DIV2K_vol_2D':
        return DIV2K(
            root=config['data_root'],
            split=split,
            scale=config['scale'],
            file_extension=config['file_extension'],
            volume_depth=config['volume_depth']
            # max_samples=config['max_samples'],
            # patch_size=config['patch_size'],
            # patch_stride=config['patch_stride']
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")