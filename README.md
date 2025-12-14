# Lung Cancer Detection using 3D CNN

This project implements a 3D Convolutional Neural Network (CNN) to detect lung nodules in CT scans using the LUNA16 dataset.

## Dataset

The project uses the **LUNA16 (LUng Nodule Analysis 2016)** dataset, which is a subset of the LIDC-IDRI dataset.

**Dataset Link:** [https://zenodo.org/records/3723295](https://zenodo.org/records/3723295)

### Download Instructions

1.  Go to the Zenodo link above.
2.  **Download `candidates.csv` and `annotations.csv` separately.** These are NOT included inside `subset0.zip`. You must download them individually from the file list on the Zenodo page.
3.  Download `subset0.zip` (this project is configured to use subset0).
4.  Extract `subset0.zip` into `data/` so you have a `data/subset0/` folder.
5.  Place `candidates.csv` and `annotations.csv` directly in the `data/` folder.

**Expected Directory Structure:**

```
data/
  candidates.csv
  annotations.csv
  subset0/
    1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031923182663544332917.mhd
    1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031923182663544332917.raw
    ...
```

## Setup

1.  Create a virtual environment (optional but recommended).
2.  Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Preprocessing (Recommended)

Preprocessing extracts 3D patches from the raw CT scans and saves them to disk. This significantly speeds up training because the model doesn't need to read and process large `.mhd` files on the fly.

To run preprocessing:

```bash
python src/preprocess.py
```

This will create a `data/processed/` directory containing:
*   `patches/`: Folder with `.npy` files (extracted 3D patches).
*   `metadata.csv`: CSV file linking patches to their labels.

### 2. Training

To train the model using the preprocessed data:

```bash
python src/train.py
```

*   The script is configured to look for data in `data/processed` by default.
*   Checkpoints will be saved as `model_epoch_X.pth`.

### 3. Visualization & Exploration

Use the Jupyter notebook to explore the data and visualize the model's input:

```bash
jupyter notebook notebooks/exploration.ipynb
```

### 4. Evaluation

To evaluate the trained model on the dataset and calculate metrics like Accuracy, Precision, Recall, and F1-Score:

```bash
python src/evaluate.py
```

## Code Structure

*   `src/dataset.py`: `LunaDataset` (raw data) and `ProcessedLunaDataset` (preprocessed data) classes.
*   `src/preprocess.py`: Script to extract patches and generate metadata.
*   `src/model.py`: `Simple3DCNN` architecture.
*   `src/train.py`: Training loop and configuration.
*   `src/evaluate.py`: Evaluation script.
*   `src/utils.py`: Helper functions for visualization.

## Improvements Implemented

*   **Data Augmentation:** Random rotation and flipping are now applied during training to improve model generalization.
*   **Evaluation Script:** A dedicated script `src/evaluate.py` is provided to assess model performance.
