import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ProcessedLunaDataset
from model import Simple3DCNN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import numpy as np

def evaluate(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Dataset
    # Ideally, you should split your dataset into train/val/test.
    # For this example, we'll just evaluate on the same dataset (or a subset if you implement splitting)
    if os.path.exists(config['processed_dir']):
        dataset = ProcessedLunaDataset(processed_dir=config['processed_dir'])
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    else:
        print("Processed data not found. Please run preprocessing first.")
        return

    # Load Model
    model = Simple3DCNN().to(device)
    model_path = config['model_path']
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model checkpoint not found at {model_path}")
        return

    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Starting evaluation...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            outputs = model(inputs)
            preds = (outputs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    print("\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    config = {
        'processed_dir': 'data/processed',
        'model_path': 'model_epoch_5.pth', # Change to your best model checkpoint
        'batch_size': 8
    }
    evaluate(config)
