import matplotlib.pyplot as plt
import numpy as np

def plot_3d_scan(scan_array, slice_idx=None):
    """
    Plots a slice of a 3D scan.
    Args:
        scan_array (numpy.ndarray): 3D array (Z, Y, X)
        slice_idx (int): Index of the slice to plot. If None, plots the middle slice.
    """
    if slice_idx is None:
        slice_idx = scan_array.shape[0] // 2
        
    plt.figure(figsize=(6, 6))
    plt.imshow(scan_array[slice_idx], cmap='gray')
    plt.title(f"Slice {slice_idx}")
    plt.axis('off')
    plt.show()

def plot_nodule(scan_array, center_xyz, spacing, origin, patch_size=64):
    """
    Extracts and plots a patch around a nodule.
    This is a visualization helper that duplicates some logic from dataset.py for standalone use.
    """
    # Convert world to voxel
    center_voxel = np.round((center_xyz - origin) / spacing).astype(int)
    
    # Simple slicing (no padding handling here for brevity in visualization)
    z, y, x = center_voxel[::-1] # numpy is z, y, x
    r = patch_size // 2
    
    patch = scan_array[z-r:z+r, y-r:y+r, x-r:x+r]
    
    plot_3d_scan(patch)
