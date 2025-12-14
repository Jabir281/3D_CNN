import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import LunaDataset, ProcessedLunaDataset
from model import Simple3DCNN
from tqdm import tqdm
import os

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = config.get('batch_size', 4)
    learning_rate = config.get('learning_rate', 0.001)
    num_epochs = config.get('num_epochs', 10)
    use_processed = config.get('use_processed', False)
    
    # Dataset and DataLoader
    if use_processed and os.path.exists(config['processed_dir']):
        print(f"Loading processed data from {config['processed_dir']}...")
        train_dataset = ProcessedLunaDataset(
            processed_dir=config['processed_dir']
        )
    else:
        print("Loading raw data (this might be slow)...")
        train_dataset = LunaDataset(
            root_dir=config['data_dir'],
            candidates_file=config['candidates_file'],
            annotations_file=config.get('annotations_file'),
            patch_size=(64, 64, 64)
        )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Model
    model = Simple3DCNN().to(device)
    
    # Loss and Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
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
            
        print(f"Epoch {epoch+1} finished. Average Loss: {running_loss / len(train_loader)}")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    # Configuration
    # Update these paths to where you extracted the LUNA16 data
    config = {
        'data_dir': 'data/subset0', # Example: path to subset0 folder
        'candidates_file': 'data/candidates.csv',
        'annotations_file': 'data/annotations.csv',
        'processed_dir': 'data/processed',
        'use_processed': True, # Set to True to use preprocessed data
        'batch_size': 4,
        'learning_rate': 1e-4,
        'num_epochs': 5
    }
    
    # Check if data exists before running
    if config['use_processed'] and os.path.exists(config['processed_dir']):
        train(config)
    elif not os.path.exists(config['data_dir']) or not os.path.exists(config['candidates_file']):
        print("Data not found. Please download the LUNA16 dataset and update the config paths.")
        print(f"Looking for data in: {config['data_dir']}")
    else:
        train(config)
