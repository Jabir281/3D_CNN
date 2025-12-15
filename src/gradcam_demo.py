"""Simple demo to run Grad-CAM on a preprocessed patch and visualize the middle slice.

Usage:
    python src/gradcam_demo.py --model model_epoch_5.pth --index 0

This script assumes you have run preprocessing and have `data/processed/metadata.csv` and
`data/processed/patches/*.npy` available.
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from dataset import ProcessedLunaDataset
from model import Simple3DCNN
from gradcam import GradCAM


def load_sample(processed_dir, index=0):
    ds = ProcessedLunaDataset(processed_dir, augment=False)
    patch, label = ds[index]
    return patch, int(label.item()), ds.metadata.iloc[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--processed_dir', default='data/processed', help='Processed data directory')
    parser.add_argument('--index', type=int, default=0, help='Index of sample to visualize')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = Simple3DCNN().to(device)
    if not os.path.exists(args.model):
        print(f"Model checkpoint not found: {args.model}")
        return
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # Load sample
    patch_tensor, label, meta = load_sample(args.processed_dir, args.index)
    print(f"Sample metadata: {meta.to_dict()}")
    print(f"Label: {label}")

    # Make batch
    input_tensor = patch_tensor.unsqueeze(0)  # (1,1,D,H,W)

    # Grad-CAM
    gradcam = GradCAM(model, target_layer_name='conv4')
    cams = gradcam.generate_cam(input_tensor, target_class=1)
    gradcam.close()

    cam = cams[0]  # (D,H,W)
    patch = patch_tensor.numpy()[0]

    # Middle slice
    z = cam.shape[0] // 2

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(patch[z], cmap='gray')
    ax.imshow(cam[z], cmap='jet', alpha=0.5)
    ax.set_title(f'Grad-CAM overlay (index={args.index}, label={label})')
    ax.axis('off')

    out_path = f'gradcam_{args.index}.png'
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved Grad-CAM overlay to {out_path}")


if __name__ == '__main__':
    main()
