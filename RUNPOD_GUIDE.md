# RunPod Setup Guide for 3D CNN Lung Cancer Detection

This guide explains how to set up and run the project on a RunPod GPU instance (e.g., A100 80GB).

## 1. Connect to RunPod

1.  Start your Pod.
2.  Click **Connect** -> **Start Web Terminal** (or use SSH if you prefer).

## 2. Clone the Repository

In the terminal, run:

```bash
git clone https://github.com/Jabir281/3D_CNN.git
cd 3D_CNN
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Download Data (The Tricky Part)

Since the data is on Zenodo, you can download it directly to the cloud instance using `wget`.

**Create the data directory:**
```bash
mkdir -p data/subset0
```

**Download `candidates.csv`:**
```bash
wget -O data/candidates.csv https://zenodo.org/records/3723295/files/candidates.csv?download=1
```

**Download `annotations.csv`:**
```bash
wget -O data/annotations.csv https://zenodo.org/records/3723295/files/annotations.csv?download=1
```

**Download `subset0.zip` (6.8 GB):**
```bash
wget -O subset0.zip https://zenodo.org/records/3723295/files/subset0.zip?download=1
```

**Extract `subset0.zip`:**
```bash
unzip subset0.zip -d data/
# This might extract into data/subset0/ or data/subset0/subset0/ depending on the zip structure.
# Check with: ls data/subset0
# If you see .mhd files directly in data/subset0, you are good.
```

*Note: If `unzip` is not installed, run `apt-get update && apt-get install unzip`.*

## 5. Preprocess Data (Parallelized)

We have optimized `src/preprocess.py` to use all available CPU cores (12 vCPUs on your instance). This should be much faster than 5 hours.

```bash
python src/preprocess.py
```

This will create `data/processed/` with `.npy` files.

## 6. Train with High Batch Size

Now you can leverage the 80GB VRAM of the A100.

*   **Batch Size:** Try `128` or `256`.
*   **Num Workers:** Set to `8` or `12` to feed the GPU fast enough.

```bash
python src/train.py --batch_size 128 --epochs 20 --num_workers 8
```

## 7. Monitor Training

You should see the progress bar moving much faster. If it's still slow, increase `num_workers` or check if the CPU is the bottleneck (run `htop` in another terminal).

## 8. Download Results

After training, you can download the `model_epoch_X.pth` files or the `results/` folder using the Jupyter Lab file browser (Right-click -> Download).
