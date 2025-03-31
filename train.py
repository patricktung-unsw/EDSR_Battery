import argparse
import torch
import yaml
from models.model_factory import get_model # type: ignore
# from datasets import data_loader
from datasets.get_dataset import get_dataset
from datasets.get_dataloader import get_loader
# import datasets
# from trainers.trainer import Trainer
from trainers.engine import Trainer
from config.default_config import default_config
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Train super-resolution models')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    # Add more command-line arguments as needed
    return parser.parse_args()

def main():
    args = parse_args()

    # Load configuration
    config = default_config.copy()
    if args.config:
        with open(args.config, 'r') as f:
            config.update(yaml.safe_load(f))
    print("Configuration loaded:", config)

    # Create save directory if it doesn't exist
    if not os.path.exists(config['save_dir']):
        os.makedirs(config['save_dir'])
    print(f"Save directory: {config['save_dir']}")

    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    config['device'] = device
    print(f"Using device: {device}")

    # Create datasets
    train_dataset = get_dataset(config['dataset'], split='train', config=config)
    val_dataset = get_dataset(config['dataset'], split='valid', config=config)
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(val_dataset)} samples")

    # # Create data loaders
    # train_loader = get_loader(train_dataset, config)
    # val_loader = get_loader(val_dataset, config)
    # # print shape of train_loader and val_loader
    # print(f"Train loader: {len(train_loader)} batches")
    # print(f"Validation loader: {len(val_loader)} batches")

    # Create model
    model = get_model(config['model'], config)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created: {config['model']}, Number of parameters: {num_params}")

    # Create trainer
    trainer = Trainer(model, train_dataset, val_dataset, config)
    # print an attribute of the trainer
    print(f"Trainer device: {trainer.device}")

    # Train the model
    trainer.train()

    print("Finished training")

if __name__ == '__main__':
    main()