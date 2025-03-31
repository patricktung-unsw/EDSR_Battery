# This script defines a Trainer class for training and validating a deep learning model. Made by Github Copilot

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.metrics import psnr
from utils.loss_factory import get_loss_function
from utils.optimizer_factory import get_optimizer
import os
import time
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, config):
        self.model = model
        self.config = config
        self.device = config['device']

        # Move model to device
        self.model.to(self.device)

        # Set up dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory']
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,  # For validation, process one image at a time
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory']
        )

        # Set up loss function and optimizer
        self.criterion = get_loss_function(config['loss_function'])
        # self.optimizer = torch.optim.Adam(
        #     model.parameters(),
        #     lr=config['learning_rate'],
        #     betas=(0.9, 0.999),
        #     eps=1e-8
        # )
        self.optimizer = get_optimizer(config['optimizer'], self.model, config)

        # Set up learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience=config['scheduler_patience'],
            factor=config['scheduler_factor']
        )

        # Initialize tracking variables
        self.best_val_loss = float('inf')
        self.current_epoch = 0

    def train_epoch(self):
        """Train the model for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_psnr = 0.0
        num_batches = len(self.train_loader)

        # Use tqdm for a progress bar
        pbar = tqdm(enumerate(self.train_loader), total=num_batches, desc=f"Epoch {self.current_epoch + 1}/{self.config['num_epochs']}")

        for batch_idx, (src, tgt) in pbar:
            # Move data to device
            src, tgt = src.to(self.device), tgt.to(self.device)

            # Zero the gradients
            self.optimizer.zero_grad()

            # Forward pass
            pred = self.model(src)

            # Calculate loss
            loss = self.criterion(pred, tgt)

            # Backward pass
            loss.backward()

            # Update weights
            self.optimizer.step()

            # Calculate PSNR
            batch_psnr = psnr(pred, tgt)

            # Update running statistics
            total_loss += loss.item()
            total_psnr += batch_psnr

            # Update progress bar
            pbar.set_postfix({'loss': loss.item(), 'psnr': batch_psnr})

        # Calculate average loss and PSNR for the epoch
        avg_loss = total_loss / num_batches
        avg_psnr = total_psnr / num_batches

        return avg_loss, avg_psnr

    def validate(self):
        """Validate the model on the validation dataset"""
        self.model.eval()
        total_loss = 0.0
        total_psnr = 0.0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for src, tgt in tqdm(self.val_loader, desc="Validating"):
                # Move data to device
                src, tgt = src.to(self.device), tgt.to(self.device)

                # Forward pass
                pred = self.model(src)

                # Calculate loss
                loss = self.criterion(pred, tgt)

                # Calculate PSNR
                batch_psnr = psnr(pred, tgt)

                # Update running statistics
                total_loss += loss.item()
                total_psnr += batch_psnr

        # Calculate average loss and PSNR for validation
        avg_loss = total_loss / num_batches
        avg_psnr = total_psnr / num_batches

        return avg_loss, avg_psnr

    def train(self):
        """Train the model for the specified number of epochs"""
        print(f"Starting training for {self.config['num_epochs']} epochs")

        for epoch in range(self.current_epoch, self.config['num_epochs']):
            self.current_epoch = epoch

            # Train for one epoch
            start_time = time.time()
            train_loss, train_psnr = self.train_epoch()
            epoch_time = time.time() - start_time

            # Validate model
            val_loss, val_psnr = self.validate()

            # Update learning rate scheduler
            self.scheduler.step(val_loss)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Print epoch statistics
            print(f"Epoch {epoch+1}/{self.config['num_epochs']} | "
                  f"Time: {epoch_time:.2f}s | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Train PSNR: {train_psnr:.2f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val PSNR: {val_psnr:.2f} | "
                  f"LR: {current_lr:.6f}")

            # Save checkpoint if this is the best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(is_best=True)
                print(f"New best model saved (Val Loss: {val_loss:.4f})")

            # Save regular checkpoint if needed
            if (epoch + 1) % self.config['checkpoint_frequency'] == 0:
                self.save_checkpoint()
                print(f"Checkpoint saved at epoch {epoch+1}")

        print(f"Training completed. Best validation loss: {self.best_val_loss:.4f}")

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        # Create directory if it doesn't exist
        os.makedirs(self.config['save_dir'], exist_ok=True)

        # Save checkpoint
        if is_best:
            checkpoint_path = os.path.join(self.config['save_dir'], f"best_model_x{self.config['scale']}.pth")
        else:
            checkpoint_path = os.path.join(self.config['save_dir'], f"checkpoint_epoch{self.current_epoch+1}_x{self.config['scale']}.pth")

        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, path):
        """Load model checkpoint"""
        if not os.path.exists(path):
            print(f"Checkpoint not found at: {path}")
            return False

        checkpoint = torch.load(path, map_location=self.device)

        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Optionally load optimizer and scheduler states
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load other training state
        if 'epoch' in checkpoint:
            self.current_epoch = checkpoint['epoch'] + 1  # Start from the next epoch

        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']

        print(f"Checkpoint loaded successfully from: {path}")
        print(f"Resuming from epoch {self.current_epoch}")

        return True