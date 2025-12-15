import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from dataset import ProcessedLunaDataset
from model import Simple3DCNN
from gradcam import GradCAM
from tqdm import tqdm

def save_gradcam_plot(patch, cam, label, pred, index, output_dir, prefix=""):
    """Saves a plot of the middle slice with Grad-CAM overlay."""
    z = patch.shape[0] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original Image
    axes[0].imshow(patch[z], cmap='gray')
    axes[0].set_title(f"Original (Label: {label})")
    axes[0].axis('off')
    
    # Grad-CAM Heatmap
    axes[1].imshow(cam[z], cmap='jet')
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(patch[z], cmap='gray')
    axes[2].imshow(cam[z], cmap='jet', alpha=0.5)
    axes[2].set_title(f"Overlay (Pred: {pred:.2f})")
    axes[2].axis('off')
    
    filename = f"{prefix}sample_{index}_label_{int(label)}_pred_{pred:.2f}.png"
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    plt.close(fig)

def visualize_results(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Dataset
    if not os.path.exists(config['processed_dir']):
        print("Processed data not found.")
        return
        
    dataset = ProcessedLunaDataset(processed_dir=config['processed_dir'], augment=False)
    # Use a smaller batch size for visualization or just iterate one by one
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Load Model
    model = Simple3DCNN().to(device)
    if not os.path.exists(config['model_path']):
        print(f"Model checkpoint not found at {config['model_path']}")
        return
        
    model.load_state_dict(torch.load(config['model_path'], map_location=device))
    model.eval()
    
    # Initialize Grad-CAM
    # Ensure 'conv4' matches the last convolutional layer in your model.py
    gradcam = GradCAM(model, target_layer_name='conv4')
    
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating visualizations in {output_dir}...")
    
    # Counters to limit number of images
    counts = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    limit = 5 # Save 5 images per category
    
    for i, (inputs, labels) in enumerate(tqdm(dataloader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        pred_prob = outputs.item()
        pred_label = 1 if pred_prob > 0.5 else 0
        true_label = int(labels.item())
        
        # Determine category
        if true_label == 1 and pred_label == 1:
            category = 'TP'
        elif true_label == 0 and pred_label == 0:
            category = 'TN'
        elif true_label == 0 and pred_label == 1:
            category = 'FP'
        elif true_label == 1 and pred_label == 0:
            category = 'FN'
            
        # Check if we need more samples for this category
        if counts[category] < limit:
            # Generate Grad-CAM
            # We want to see why it predicted positive (target_class=1) 
            # or why it predicted negative (target_class=0)
            # Usually for medical imaging, we are interested in the positive class activation
            cams = gradcam.generate_cam(inputs, target_class=1)
            
            patch = inputs.cpu().numpy()[0, 0] # (D, H, W)
            cam = cams[0]
            
            save_gradcam_plot(patch, cam, true_label, pred_prob, i, output_dir, prefix=f"{category}_")
            counts[category] += 1
            
        # Stop if we have enough of everything
        if all(c >= limit for c in counts.values()):
            break
            
    gradcam.close()
    print("Visualization complete.")

if __name__ == "__main__":
    config = {
        'processed_dir': 'data/processed',
        'model_path': 'model_epoch_5.pth', # Update with your best model
        'output_dir': 'results/visualizations'
    }
    visualize_results(config)
