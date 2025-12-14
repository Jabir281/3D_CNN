# Lung Cancer Detection using 3D CNN

This project implements a 3D Convolutional Neural Network (CNN) to detect lung nodules in CT scans using the LUNA16 dataset.

## Dataset

The project uses the **LUNA16 (LUng Nodule Analysis 2016)** dataset, which is a subset of the LIDC-IDRI dataset.

**Dataset Link:** [https://zenodo.org/records/3723295](https://zenodo.org/records/3723295)

### Download Instructions

1.  Go to the Zenodo link above.
2.  Download `candidates.csv` and `annotations.csv`.
3.  Download `subset0.zip` (this project is configured to use subset0).
4.  Extract the files into the `data/` directory.

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

### Training

To train the model, run:

```bash
python src/train.py
```

Make sure to update the `config` dictionary in `src/train.py` if your data is located elsewhere.

### Code Structure

*   `src/dataset.py`: `LunaDataset` class for loading CT scans and extracting 3D patches.
*   `src/model.py`: `Simple3DCNN` architecture.
*   `src/train.py`: Training loop and configuration.
*   `src/utils.py`: Helper functions for visualization.

## Notes

*   **Preprocessing:** The current implementation performs on-the-fly patch extraction and normalization. For large-scale training, it is recommended to pre-process the data (resample to 1mm spacing, extract patches) and save them to disk to speed up training.
*   **Model:** The provided model is a simple baseline. State-of-the-art approaches often use 3D ResNets, U-Nets (for segmentation), or attention mechanisms.
