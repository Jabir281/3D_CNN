import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
import random
from scipy.ndimage import rotate

class LunaDataset(Dataset):
    def __init__(self, root_dir, candidates_file, annotations_file=None, subset_indices=None, patch_size=(64, 64, 64)):
        """
        Args:
            root_dir (string): Directory with all the images (e.g. data/subset0, data/subset1...).
                               Since LUNA16 is split into subsets, this might need to handle multiple folders.
                               For simplicity, we assume all .mhd files are discoverable or we pass a list of paths.
            candidates_file (string): Path to the candidates.csv file.
            annotations_file (string): Path to the annotations.csv file.
            subset_indices (list): List of subset indices to include (0-9).
            patch_size (tuple): Size of the 3D patch to extract.
        """
        self.root_dir = root_dir
        self.patch_size = patch_size
        
        # Load candidates
        self.candidates = pd.read_csv(candidates_file)
        
        # Filter candidates to only include those present in the root_dir (subset0)
        self.candidates = self._filter_candidates(self.candidates)
        print(f"Filtered candidates to {len(self.candidates)} records found in {self.root_dir}")

        # Load annotations if provided (for training labels)
        self.labels = {}
        if annotations_file:
            annotations = pd.read_csv(annotations_file)
            # Create a dictionary or set for fast lookup of positive nodules
            # This is a simplification. In reality, you match by distance.
            # For this template, we will rely on the 'class' column in candidates.csv if it exists,
            # or assume candidates.csv is the 'candidates_V2.csv' which has the class column?
            # Actually, standard candidates.csv has 'class' column (0 or 1).
            pass

        # If candidates.csv has a 'class' column, use it.
        if 'class' in self.candidates.columns:
            self.has_labels = True
        else:
            self.has_labels = False

    def _filter_candidates(self, candidates):
        """
        Filters the candidates DataFrame to only include series_uids that exist in the root_dir.
        """
        existing_series_uids = set()
        for filename in os.listdir(self.root_dir):
            if filename.endswith(".mhd"):
                existing_series_uids.add(filename[:-4]) # Remove .mhd extension
        
        return candidates[candidates['seriesuid'].isin(existing_series_uids)]

    def __len__(self):
        return len(self.candidates)

    def __getitem__(self, idx):
        row = self.candidates.iloc[idx]
        series_uid = row['seriesuid']
        center_xyz = (row['coordX'], row['coordY'], row['coordZ'])
        
        if self.has_labels:
            label = int(row['class'])
        else:
            label = 0 # Dummy label for inference

        # Find the image file
        # We search recursively in root_dir for {series_uid}.mhd
        image_path = self._find_image_file(series_uid)
        
        if image_path is None:
            # Skip or return error. For training stability, maybe return a zero tensor?
            # But better to fail loud so user knows data is missing.
            raise FileNotFoundError(f"Image for series UID {series_uid} not found in {self.root_dir}")

        # Load image
        itk_image = sitk.ReadImage(image_path)
        
        # Extract patch
        patch = self._extract_patch(itk_image, center_xyz, self.patch_size)
        
        # Normalize
        patch = self._normalize(patch)
        
        # Convert to tensor (C, D, H, W)
        patch_tensor = torch.from_numpy(patch).float().unsqueeze(0)
        
        return patch_tensor, torch.tensor(label, dtype=torch.float32)

    def _find_image_file(self, series_uid):
        # Simple search. In production, cache this mapping.
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file == f"{series_uid}.mhd":
                    return os.path.join(root, file)
        return None

    def _extract_patch(self, itk_image, center_xyz, patch_size):
        # Convert world coordinates to voxel coordinates
        origin = np.array(itk_image.GetOrigin())
        spacing = np.array(itk_image.GetSpacing())
        
        center_voxel = np.round((center_xyz - origin) / spacing).astype(int)
        
        # Calculate start and end voxel indices
        start_index = center_voxel - np.array(patch_size) // 2
        end_index = start_index + np.array(patch_size)
        
        # Handle boundary conditions (padding)
        image_array = sitk.GetArrayFromImage(itk_image) # (Z, Y, X) order
        
        # Note: SimpleITK uses (X, Y, Z) for coordinates, but numpy uses (Z, Y, X)
        # We need to be careful with axes.
        # center_voxel is (X, Y, Z)
        # patch_size is (X, Y, Z)
        
        # Transpose to (Z, Y, X) for numpy slicing
        start_index_zxy = start_index[::-1]
        end_index_zxy = end_index[::-1]
        patch_size_zxy = np.array(patch_size)[::-1]
        
        # Pad image if necessary
        pad_before = np.maximum(-start_index_zxy, 0)
        pad_after = np.maximum(end_index_zxy - np.array(image_array.shape), 0)
        
        if np.any(pad_before > 0) or np.any(pad_after > 0):
            image_array = np.pad(image_array, 
                                 list(zip(pad_before, pad_after)), 
                                 mode='constant', 
                                 constant_values=-1000) # Air density
            
            # Adjust indices after padding
            start_index_zxy += pad_before
            end_index_zxy += pad_before

        patch = image_array[start_index_zxy[0]:end_index_zxy[0],
                            start_index_zxy[1]:end_index_zxy[1],
                            start_index_zxy[2]:end_index_zxy[2]]
        
        return patch

    def _normalize(self, patch):
        # Hounsfield Unit normalization
        min_hu = -1000
        max_hu = 400
        patch = np.clip(patch, min_hu, max_hu)
        patch = (patch - min_hu) / (max_hu - min_hu)
        return patch

class ProcessedLunaDataset(Dataset):
    def __init__(self, processed_dir, transform=None, augment=False):
        """
        Args:
            processed_dir (string): Directory containing 'metadata.csv' and 'patches/' folder.
            transform (callable, optional): Optional transform to be applied on a sample.
            augment (bool): Whether to apply data augmentation.
        """
        self.processed_dir = processed_dir
        self.metadata_file = os.path.join(processed_dir, 'metadata.csv')
        self.augment = augment
        
        if not os.path.exists(self.metadata_file):
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_file}. Run preprocess.py first.")
            
        self.metadata = pd.read_csv(self.metadata_file)
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        filename = row['filename']
        label = row['label']
        
        file_path = os.path.join(self.processed_dir, 'patches', filename)
        
        # Load patch
        patch = np.load(file_path)
        
        if self.augment:
            patch = self._augment(patch)
        
        # Convert to tensor (C, D, H, W)
        # patch is (64, 64, 64) -> (1, 64, 64, 64)
        patch_tensor = torch.from_numpy(patch.copy()).float().unsqueeze(0)
        
        return patch_tensor, torch.tensor(label, dtype=torch.float32)

    def _augment(self, patch):
        # Random rotation
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            axes = random.choice([(0, 1), (1, 2), (0, 2)])
            patch = rotate(patch, angle, axes=axes, reshape=False)
            
        # Random flip
        if random.random() > 0.5:
            axis = random.choice([0, 1, 2])
            patch = np.flip(patch, axis=axis)
            
        return patch
