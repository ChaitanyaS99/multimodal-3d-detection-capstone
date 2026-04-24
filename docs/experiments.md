# Experimental Catalog

This document catalogs the experimental configurations explored during the project. Each experiment represents a deliberate hypothesis about how to improve multi-modal 3D object detection performance under the constraints of the nuScenes mini subset.

The experiments are organized into four categories: **baselines**, **model optimization**, **data strategies**, and **inference & efficiency**.

---

## Baselines

### 1. LiDAR-Only Baseline
**File:** `configs/transfusion_nusc_voxel_L.py`

The reference single-modal configuration using only LiDAR point clouds. Establishes the geometric-only performance ceiling without camera input. Trained for 20 epochs with cyclic LR scheduling.

### 2. Multi-Modal Baseline
**File:** `configs/transfusion_nusc_voxel_LC.py`

The full LiDAR + Camera configuration with cross-attention fusion via the TransFusionHead. Image backbone (ResNet-50) is frozen at the first stage. Trained for 6 epochs as the canonical multi-modal reference point for all subsequent experiments.

---

## Model Optimization Experiments

### 3. Knowledge Distillation
**File:** `configs/transfusion_kd.py`

Tests softer training dynamics under teacher guidance. Reduces Focal Loss gamma from 2.0 to 1.5, lowers learning rate by half, and loads from a pre-trained checkpoint. Hypothesis: gentler training reduces overfitting on the small dataset.

### 4. Transfer Learning with Frozen Layers
**File:** `configs/transfusion_PRETRAINED.py`

Freezes early stages of both image and LiDAR backbones to leverage pre-trained features. Uses an ultra-low learning rate (1e-5). Hypothesis: only the fusion head needs to adapt to the dataset; backbone features should remain stable.

### 5. Aggressive Class Imbalance Handling
**File:** `configs/transfusion_ULTIMATE.py`

Applies high-weight Focal Loss (loss_weight=4.0), conservative learning rate (5e-5), and a stable step schedule over 35 epochs. Hypothesis: severe class imbalance is the primary bottleneck and requires aggressive loss reweighting.

### 6. LiDAR-Primary with Adaptive Camera
**File:** `configs/transfusion_MAXIMUM_PERFORMANCE.py`

Restructures the model to make LiDAR the primary modality and camera a helper. Uses a smaller ResNet-18 image backbone with frozen early layers, optimized NMS thresholds, and slow camera-side learning. Hypothesis: under data constraints, camera features add noise; LiDAR-primary fusion is more stable.

### 7. Frozen Backbone with Minimal Decoder
**File:** `configs/transfusion_MULTIMODAL_SMALL.py`

Freezes both backbones almost entirely and uses a minimal 2-layer decoder with reduced attention heads. Trains only the classification head with very high loss weight (8.0). Hypothesis: with such limited data, the only trainable component should be the final classification layer.

---

## Data Strategy Experiments

### 8. Aggressive Data Augmentation
**File:** `configs/transfusion_AUGMENTED.py`

Applies heavy augmentation (full-range rotations, scale 0.7–1.3, translation up to 2m, photometric distortion) over 80 epochs with frozen backbones. Hypothesis: artificial data multiplication can compensate for the small dataset size.

### 9. Copy-Paste Augmentation with Dataset Repetition
**File:** `configs/transfusion_copy_paste_final.py`

Combines aggressive augmentation with `RepeatDataset` (×3 multiplication), large batch size (6), 40 epochs of training, FP16 mixed-precision, and cosine annealing. Hypothesis: dataset repetition combined with strong augmentation effectively expands the training distribution.

### 10. Multi-Modal Pipeline Variant
**File:** `configs/multimodal_transfusion.py`

Implements an alternative full multi-modal pipeline with a 2-layer transformer decoder, FP16 mixed precision, cosine annealing schedule, and 26 training epochs. Tests whether deeper fusion (2 decoder layers) improves results.

---

## Inference & Efficiency Experiments

### 11. Multi-GPU High-Throughput Training
**File:** `configs/nuclear_power.py`

Maximizes GPU utilization with 8 samples per GPU, higher learning rate (1e-3), and 20 epochs of training. Tests whether stronger compute alone yields better convergence.

---

## Additional Experiments (40+ Variants)

Beyond the 11 primary configurations above, the project explored a broad set of additional variants captured in the original TransFusion configs directory, including:

| Variant Family | Strategies Tested |
|---|---|
| Stage-wise training | `STAGE1`, `STAGE2`, `STAGE3` — progressive training stages |
| Class-focused | `2CLASS_FOCUS`, `FIXED_CLASSES`, `PERFECT_CLASSES` — narrowed class scope |
| Schedule variants | `6epochs`, `10epochs`, `single_gpu_aggressive` — training duration sweeps |
| TTA variants | `tta_boost`, `working_tta`, `immediate_tta_boost` — test-time augmentation |
| Copy-paste variants | `copy_paste_compatible`, `copy_paste_no_fp16`, `copy_paste_manual` — augmentation refinements |
| Architecture sweeps | `SIMPLE`, `SIMPLE_FUSION`, `SIMPLE_OPTIMIZED`, `PROGRESSIVE_COMPLEXITY` — complexity ablations |

A summary of the work directories produced by these experiments is available in the project's experiment logs.

---

## Cross-Cutting Findings

The structured experimental study revealed several consistent patterns:

- **Performance stability across optimization strategies.** Across all 11+ primary variants, multi-modal performance remained within a narrow band, with marginal variations from individual techniques.
- **Limited gains from isolated optimization.** Knowledge distillation, loss tuning, and ensembling each produced small improvements but none dramatically shifted the performance ceiling.
- **Class frequency dominates results.** Strong, reliable detection on cars and pedestrians; reduced accuracy on rare categories regardless of optimization strategy.
- **Fusion behavior is governed by representation quality.** The pattern of stable performance across optimization choices suggests bottlenecks are in feature alignment and cross-modal representation, not in training tricks.

These findings motivate the project's research direction toward **physics-informed, geometry-aware multimodal perception** — see [`results.md`](results.md) for the full discussion.

---

## Running an Experiment

To reproduce any of the experiments above:

```bash
# Train
python tools/train.py configs/<config_name>.py \
    --work-dir work_dirs/<run_name>

# Evaluate
python tools/test.py configs/<config_name>.py \
    work_dirs/<run_name>/epoch_N.pth --eval bbox
```

See [`setup.md`](setup.md) for environment setup.
