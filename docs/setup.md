# Setup & Installation Guide

This document describes how to set up the environment, install dependencies, and prepare the nuScenes dataset for running the experiments in this repository.

## System Requirements

- **OS:** Linux (Ubuntu 20.04+ recommended)
- **Python:** 3.7+
- **CUDA:** 10.2 or 11.x
- **GPU:** NVIDIA GPU with at least 11GB VRAM for training (project tested on RTX A6000 with 47GB)
- **Disk:** ~500 GB free space recommended for full nuScenes dataset; ~5 GB sufficient for the mini subset

## Installation

### Step 1 — Create a Conda environment

```bash
conda create -n transfusion python=3.7 -y
conda activate transfusion
```

### Step 2 — Install PyTorch (with CUDA support)

```bash
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch
```

### Step 3 — Install MMCV and MMDetection

```bash
pip install mmcv-full==1.2.4 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.1/index.html
pip install mmdet==2.10.0
```

### Step 4 — Install MMDetection3D (from the included build)

```bash
cd mmdetection3d
pip install -v -e .
```

### Step 5 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

### Step 6 — Verify installation

```python
import mmcv, mmdet, mmdet3d
print(mmcv.__version__, mmdet.__version__, mmdet3d.__version__)
```

## Dataset Preparation

### Download the nuScenes dataset

Visit [nuscenes.org](https://www.nuscenes.org/nuscenes) and register for an account. Download either:

- **nuScenes mini** (10 scenes, ~5 GB) — used for the experiments documented here
- **nuScenes trainval** (full dataset, ~350 GB) — required to scale results to the published TransFusion baseline

### Organize the data

Place the downloaded data under `data/nuscenes/` following this structure:

```
data/nuscenes/
├── maps/
├── samples/
├── sweeps/
├── v1.0-mini/ (or v1.0-trainval)
└── nuscenes_infos_train.pkl (generated in next step)
```

### Generate the info files

```bash
python tools/create_data.py nuscenes \
 --root-path ./data/nuscenes \
 --out-dir ./data/nuscenes \
 --extra-tag nuscenes \
 --version v1.0-mini
```

For the full dataset, replace `v1.0-mini` with `v1.0-trainval`.

This will produce:
- `nuscenes_infos_train.pkl`
- `nuscenes_infos_val.pkl`
- `nuscenes_dbinfos_train.pkl`

## Running Experiments

### Train the LiDAR-only baseline

```bash
python tools/train.py configs/transfusion_nusc_voxel_L.py --work-dir work_dirs/baseline_lidar
```

### Train the multi-modal (LiDAR + Camera) baseline

```bash
python tools/train.py configs/transfusion_nusc_voxel_LC.py --work-dir work_dirs/baseline_multimodal
```

### Evaluate a trained model

```bash
python tools/test.py \
 configs/transfusion_nusc_voxel_LC.py \
 work_dirs/baseline_multimodal/epoch_6.pth \
 --eval bbox
```

### Run an experiment variant

Each experimental configuration in `configs/` can be trained the same way. For example:

```bash
python tools/train.py configs/transfusion_kd.py --work-dir work_dirs/kd_experiment
```

See [`configs/README.md`](../configs/README.md) for a guide to all available configurations.

## Common Issues

- **CUDA out of memory:** Reduce `samples_per_gpu` in the config or freeze backbone stages
- **MMCV version mismatch:** Use `mmcv-full==1.2.4` exactly; newer versions break TransFusion compatibility
- **Coordinate system errors:** If using a newer mmdet3d version to generate info files, expect incorrect mAOE/mASE due to a coordinate refactor in the upstream repo
- **Disk quota errors during data prep:** Use a separate scratch directory for the database info files
