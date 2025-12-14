import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataset import LunaDataset
import torch

def preprocess_data(config):
    print("Starting preprocessing...")
    
    # Initialize the raw dataset
    # This handles the complex logic of reading .mhd files and extracting patches
    raw_dataset = LunaDataset(
        root_dir=config['data_dir'],
        candidates_file=config['candidates_file'],
        patch_size=(64, 64, 64)
    )
    
    output_dir = config['output_dir']
    patches_dir = os.path.join(output_dir, 'patches')
    os.makedirs(patches_dir, exist_ok=True)
    
    processed_metadata = []
    
    print(f"Found {len(raw_dataset)} candidates in {config['data_dir']}")
    print(f"Saving processed patches to {patches_dir}")
    
    # Iterate through the dataset and save patches to disk
    for i in tqdm(range(len(raw_dataset))):
        try:
            # raw_dataset[i] returns (patch_tensor, label_tensor)
            patch_tensor, label_tensor = raw_dataset[i]
            
            # Convert back to numpy for saving
            # patch_tensor is (1, 64, 64, 64) -> save as (64, 64, 64)
            patch_array = patch_tensor.numpy()[0] 
            
            # Get metadata for filename
            row = raw_dataset.candidates.iloc[i]
            series_uid = row['seriesuid']
            
            # Create a unique filename: seriesuid_index.npy
            filename = f"{series_uid}_{i}.npy"
            file_path = os.path.join(patches_dir, filename)
            
            # Save the numpy array
            np.save(file_path, patch_array)
            
            # Record metadata
            processed_metadata.append({
                'filename': filename,
                'label': int(label_tensor.item()),
                'seriesuid': series_uid,
                'original_index': i
            })
            
        except Exception as e:
            print(f"Error processing index {i}: {e}")
            continue
            
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
        'output_dir': 'data/processed'
    }
    
    # Check inputs
    if not os.path.exists(config['data_dir']):
        print(f"Error: Data directory '{config['data_dir']}' not found.")
    elif not os.path.exists(config['candidates_file']):
        print(f"Error: Candidates file '{config['candidates_file']}' not found.")
    else:
        preprocess_data(config)
