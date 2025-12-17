import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from dataset import LunaDataset, ProcessedLunaDataset
from model import Simple3DCNN
from tqdm import tqdm
import os
import glob
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import argparse

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = config.get('batch_size', 4)
    learning_rate = config.get('learning_rate', 0.001)
    num_epochs = config.get('num_epochs', 10)
    use_processed = config.get('use_processed', False)
    num_workers = config.get('num_workers', 0)
    
    print(f"Batch Size: {batch_size}, Epochs: {num_epochs}, LR: {learning_rate}")

    # Dataset and DataLoader
    if use_processed and os.path.exists(config['processed_dir']):
        print(f"Loading processed data from {config['processed_dir']}...")
        full_dataset = ProcessedLunaDataset(
            processed_dir=config['processed_dir'],
            augment=True 
        )
        
        # Stratified Split (80% Train, 20% Val)
        print("Splitting data into Train (80%) and Validation (20%)...")
        labels = full_dataset.metadata['label'].values
        indices = np.arange(len(labels))
        
        train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)
        
        # Create Subsets
        # Train set has augmentation enabled (from full_dataset)
        train_dataset = Subset(full_dataset, train_idx)
        
        # Val set should NOT have augmentation
        val_full_dataset = ProcessedLunaDataset(processed_dir=config['processed_dir'], augment=False)
        val_dataset = Subset(val_full_dataset, val_idx)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Calculate Class Weights for Weighted Loss
        n_pos = sum(labels[train_idx])
        n_neg = len(train_idx) - n_pos
        pos_weight = n_neg / max(n_pos, 1) # Avoid div by zero
        print(f"Class Imbalance Ratio: 1:{pos_weight:.1f}. Using Weighted Loss.")
        
    else:
        print("Loading raw data (this might be slow)...")
        # Fallback for raw data (no split implemented here for brevity, assuming processed is used)
        train_dataset = LunaDataset(
            root_dir=config['data_dir'],
            candidates_file=config['candidates_file'],
            annotations_file=config.get('annotations_file'),
            patch_size=(64, 64, 64)
        )
        val_dataset = None
        pos_weight = 1.0
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Model
    model = Simple3DCNN().to(device)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Check for resume
    start_epoch = 0
    if config.get('resume', False):
        checkpoints = glob.glob("results/model_epoch_*.pth")
        if checkpoints:
            # Sort by epoch number extracted from filename
            checkpoints.sort(key=lambda x: int(re.search(r'model_epoch_(\d+).pth', x).group(1)))
            latest_checkpoint = checkpoints[-1]
            print(f"Resuming from checkpoint: {latest_checkpoint}")
            
            try:
                model.load_state_dict(torch.load(latest_checkpoint, map_location=device))
                start_epoch = int(re.search(r'model_epoch_(\d+).pth', latest_checkpoint).group(1))
                print(f"Resuming training from epoch {start_epoch + 1}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Starting from scratch.")
        else:
            print("No checkpoints found in results/. Starting from scratch.")

    # Loss and Optimizer
    # Custom Weighted BCELoss
    class WeightedBCELoss(nn.Module):
        def __init__(self, pos_weight):
            super().__init__()
            self.pos_weight = torch.tensor(pos_weight).float().to(device)
            
        def forward(self, outputs, targets):
            outputs = torch.clamp(outputs, 1e-7, 1 - 1e-7)
            loss = -(self.pos_weight * targets * torch.log(outputs) + (1 - targets) * torch.log(1 - outputs))
            return loss.mean()

    criterion = WeightedBCELoss(pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training Loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1) # (Batch, 1)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': running_loss / (progress_bar.n + 1)})
            
        print(f"Epoch {epoch+1} Train Loss: {running_loss / len(train_loader):.4f}")
        
        # Validation Loop
        if val_dataset:
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                    inputs = inputs.to(device)
                    labels = labels.to(device).unsqueeze(1)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    val_preds.extend(outputs.cpu().numpy())
                    val_targets.extend(labels.cpu().numpy())
            
            val_preds = np.array(val_preds) > 0.5
            val_acc = np.mean(val_preds == np.array(val_targets))
            print(f"Epoch {epoch+1} Val Loss: {val_loss / len(val_loader):.4f} | Val Acc: {val_acc:.4f}")

        # Save checkpoint
        torch.save(model.state_dict(), f"results/model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train 3D CNN for Lung Cancer Detection')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for data loading')
    parser.add_argument('--no_resume', action='store_true', help='Do not resume from checkpoint')
    
    args = parser.parse_args()

    # Configuration
    # Update these paths to where you extracted the LUNA16 data
    config = {
        'data_dir': 'data/subset0', # Example: path to subset0 folder
        'candidates_file': 'data/candidates.csv',
        'annotations_file': 'data/annotations.csv',
        'processed_dir': 'data/processed',
        'use_processed': True, # Set to True to use preprocessed data
        'resume': not args.no_resume, # Auto-resume from latest checkpoint
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'num_epochs': args.epochs,
        'num_workers': args.num_workers
    }
    
    # Check if data exists before running
    if config['use_processed'] and os.path.exists(config['processed_dir']):
        train(config)
    elif not os.path.exists(config['data_dir']) or not os.path.exists(config['candidates_file']):
        print("Data not found. Please download the LUNA16 dataset and update the config paths.")
        print(f"Looking for data in: {config['data_dir']}")
    else:
        train(config)
