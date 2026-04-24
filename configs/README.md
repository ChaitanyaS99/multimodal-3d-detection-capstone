# Experimental Configurations

This directory contains all the TransFusion-based configuration files used in the project. Each file represents a deliberate experimental hypothesis tested during the research.

For the full experimental rationale and findings, see [`../docs/experiments.md`](../docs/experiments.md).

## File Index

### Baselines

| File | Purpose |
|---|---|
| `transfusion_nusc_voxel_L.py` | LiDAR-only TransFusion baseline (20 epochs, cyclic LR) |
| `transfusion_nusc_voxel_LC.py` | Multi-modal LiDAR + Camera baseline (6 epochs) — canonical reference for all experiments |

### Model Optimization

| File | Strategy |
|---|---|
| `transfusion_kd.py` | Softer training: lower Focal Loss gamma (1.5), half learning rate, teacher checkpoint loading |
| `transfusion_PRETRAINED.py` | Transfer learning: freeze early backbone stages, ultra-low LR (1e-5) |
| `transfusion_ULTIMATE.py` | Class imbalance fix: high-weight Focal Loss (4.0), step schedule, 35 epochs |
| `transfusion_MAXIMUM_PERFORMANCE.py` | LiDAR-primary fusion: smaller image backbone, slow camera learning, optimized NMS |
| `transfusion_MULTIMODAL_SMALL.py` | Frozen-backbone variant: minimal trainable parameters, 2-layer decoder, 60 epochs |

### Data Strategies

| File | Strategy |
|---|---|
| `transfusion_AUGMENTED.py` | Aggressive augmentation: full-range rotations, heavy scaling, 80 epochs, frozen backbones |
| `copy_paste_final.py` | Copy-Paste augmentation with `RepeatDataset` ×3, batch 6, 40 epochs, FP16, cosine annealing |
| `multimodal_transfusion.py` | Alternative full multi-modal pipeline with 2-layer decoder, FP16, cosine annealing |

### Inference & Throughput

| File | Strategy |
|---|---|
| `nuclear_power.py` | High-throughput multi-GPU training: 8 samples/GPU, high LR (1e-3), 20 epochs |

## Configuration Conventions

All configurations follow MMDetection3D's convention. Key fields to look for:

- `_base_` — parent configuration that the variant extends
- `model` — architectural overrides
- `optimizer` — training optimization settings
- `lr_config` — learning rate schedule
- `runner` / `total_epochs` — training duration
- `work_dir` — output directory for checkpoints and logs
- `load_from` — pre-trained checkpoint to initialize from (if any)

## Usage

To train any configuration:

```bash
python tools/train.py configs/<config_name>.py --work-dir work_dirs/<run_name>
```

To evaluate a trained model:

```bash
python tools/test.py configs/<config_name>.py work_dirs/<run_name>/epoch_N.pth --eval bbox
```

See [`../docs/setup.md`](../docs/setup.md) for environment setup and dataset preparation.

## A Note on Experimental Breadth

The 11 configurations included here represent the primary experimental thread. The broader research process explored 40+ additional variants across stage-wise training, class-focused experiments, schedule sweeps, test-time augmentation strategies, and architectural ablations. The core configurations preserved here are those that best document the systematic experimental approach taken.
