import os
# Set environment variables to prevent thread explosion before importing numpy/torch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from tqdm import tqdm
from dataset import LunaDataset
import torch
from multiprocessing import Pool, cpu_count
from functools import partial

def process_sample(i, dataset, patches_dir):
    try:
        # Get metadata for filename
        row = dataset.candidates.iloc[i]
        series_uid = row['seriesuid']
        
        # Create a unique filename: seriesuid_index.npy
        filename = f"{series_uid}_{i}.npy"
        file_path = os.path.join(patches_dir, filename)
        
        # Check if file already exists (Resume capability)
        if os.path.exists(file_path):
            # If it exists, we just need the metadata, we don't need to load the heavy image
            label = int(row['class']) if 'class' in row else 0
            return {
                'filename': filename,
                'label': label,
                'seriesuid': series_uid,
                'original_index': i
            }

        # raw_dataset[i] returns (patch_tensor, label_tensor)
        patch_tensor, label_tensor = dataset[i]
        
        # Convert back to numpy for saving
        # patch_tensor is (1, 64, 64, 64) -> save as (64, 64, 64)
        patch_array = patch_tensor.numpy()[0] 
        
        # Save the numpy array
        np.save(file_path, patch_array)
        
        return {
            'filename': filename,
            'label': int(label_tensor.item()),
            'seriesuid': series_uid,
            'original_index': i
        }
        
    except Exception as e:
        # print(f"Error processing index {i}: {e}")
        return None

def preprocess_data(config):
    print("Starting preprocessing...")
    
    # Initialize the raw dataset
    raw_dataset = LunaDataset(
        root_dir=config['data_dir'],
        candidates_file=config['candidates_file'],
        patch_size=(64, 64, 64)
    )
    
    output_dir = config['output_dir']
    patches_dir = os.path.join(output_dir, 'patches')
    os.makedirs(patches_dir, exist_ok=True)
    
    print(f"Found {len(raw_dataset)} candidates in {config['data_dir']}")
    print(f"Saving processed patches to {patches_dir}")
    
    # Use multiprocessing
    # Default to half the CPUs to avoid memory issues, or 4 if not specified
    default_workers = max(1, cpu_count() // 2)
    num_workers = config.get('num_workers', default_workers)
    print(f"Using {num_workers} workers for preprocessing.")
    
    # We need to pass the dataset to the worker, but passing the whole dataset object 
    # might be slow if it's large (pickling). 
    # However, LunaDataset is lightweight (just a dataframe), so it should be fine.
    # But SimpleITK objects inside might be an issue if cached. 
    # Our LunaDataset doesn't cache images, so it's safe.
    
    # Create a partial function with fixed arguments
    worker_func = partial(process_sample, dataset=raw_dataset, patches_dir=patches_dir)
    
    processed_metadata = []
    indices = list(range(len(raw_dataset)))
    
    with Pool(num_workers) as pool:
        # Use tqdm to show progress
        results = list(tqdm(pool.imap(worker_func, indices), total=len(indices)))
    
    # Filter out None results (errors)
    processed_metadata = [r for r in results if r is not None]
            
    # Save the new metadata CSV
    if processed_metadata:
        df = pd.DataFrame(processed_metadata)
        metadata_path = os.path.join(output_dir, 'metadata.csv')
        df.to_csv(metadata_path, index=False)
        print(f"Preprocessing complete. Metadata saved to {metadata_path}")
        print(f"Total processed samples: {len(df)}")
    else:
        print("No samples were processed.")

if __name__ == "__main__":
    # Configuration
    config = {
        'data_dir': 'data/subset0',
        'candidates_file': 'data/candidates.csv',
        'output_dir': 'data/processed',
        'num_workers': 6 # User requested 6 workers
    }
    
    # Check inputs
    if not os.path.exists(config['data_dir']):
        print(f"Error: Data directory '{config['data_dir']}' not found.")
    elif not os.path.exists(config['candidates_file']):
        print(f"Error: Candidates file '{config['candidates_file']}' not found.")
    else:
        preprocess_data(config)
