from torch.utils.data import DataLoader

def get_loader(dataset, config):
    """
    Create a DataLoader for the given dataset.

    Args:
        dataset: The dataset object (e.g., train_dataset or val_dataset).
        config: Configuration dictionary with parameters like batch_size, num_workers, etc.

    Returns:
        DataLoader object.
    """
    return DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True if dataset.split == 'train' else False,  # Shuffle only for training
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )