# Methodology

This document describes the research methodology, experimental design, and analytical framework used in this project.

## Research Question

**How does a TransFusion-based multimodal 3D object detection architecture behave under different optimization strategies, and where do its performance bottlenecks lie?**

This work focuses on two complementary objectives:

1. Build a complete, reproducible multimodal 3D detection pipeline based on the TransFusion architecture
2. Conduct a structured experimental study to understand how the model responds to different optimization techniques and to identify where genuine improvements require deeper architectural or representational changes

## Approach

The methodology follows a four-phase structured approach.

### Phase 1 — Baseline Establishment

The first phase establishes a reproducible baseline by training and evaluating both LiDAR-only and multi-modal TransFusion configurations on the nuScenes mini subset (323 training / 81 validation samples). This phase confirms the pipeline operates end-to-end and produces a benchmark for subsequent comparisons.

**Configurations:**
- `transfusion_nusc_voxel_L.py` — LiDAR-only baseline
- `transfusion_nusc_voxel_LC.py` — Multi-modal (LiDAR + Camera) baseline

### Phase 2 — Systematic Experimentation

The second phase explores a comprehensive suite of optimization strategies grouped into three categories:

**Model Optimization:**
- Loss rebalancing (Focal Loss tuning, class-weighted losses)
- Knowledge distillation (softer training with teacher guidance)
- Learning rate tuning across schedules (cyclic, step, cosine annealing)
- Frozen backbone strategies (transfer learning under limited data)

**Data Strategies:**
- Aggressive data augmentation (rotation, scaling, flipping, photometric)
- Multi-scale training pipelines
- Repeat-dataset strategies for effective epoch multiplication

**Inference & Efficiency:**
- FP16 quantization
- Confidence threshold tuning
- NMS parameter optimization
- Model ensembling and Test-Time Augmentation (TTA)

Each configuration variant is documented in `configs/` with a one-line summary of the strategy it tests. See [`experiments.md`](experiments.md) for the full catalog.

### Phase 3 — Performance Analysis

The third phase conducts detailed analysis of detection behavior beyond aggregate mAP scores:

- **Per-class performance breakdown** across all 10 nuScenes object categories
- **Distance-threshold analysis** measuring detection precision at varying spatial tolerances (0.5m, 1m, 2m, 4m)
- **Confidence calibration analysis** examining detection counts vs. confidence scores per class
- **Power-law scaling analysis** projecting model performance across dataset sizes

Analysis scripts are in `scripts/` and outputs in `results/`.

### Phase 4 — Data Variety Assessment

The fourth phase evaluates the suitability of the nuScenes mini subset for the research scope. This includes scoring environmental diversity (weather, lighting, road types, traffic, geography) and assessing which research questions the dataset can validly support. This step provides honest scoping for the experimental conclusions.

See `results/reports/data_variety_assessment.md` for the full assessment.

## Evaluation Metrics

| Metric | Description |
|---|---|
| **mAP** | Mean Average Precision across all 10 classes |
| **Per-class AP** | Average Precision for each individual class |
| **Distance-threshold AP** | AP at varying spatial precision (0.5m, 1m, 2m, 4m) |
| **NDS** | nuScenes Detection Score (composite metric) |
| **Detection counts** | Number of predictions per class |
| **Confidence statistics** | Maximum and high-confidence detection counts per class |

All metrics follow the standard nuScenes evaluation protocol, computed via the official nuScenes devkit.

## Hardware & Software Environment

| Component | Specification |
|---|---|
| GPU | 6 × NVIDIA RTX A6000 (47 GB VRAM each) |
| Server | University HPC cluster with shared storage |
| Framework | PyTorch 1.7.1 |
| Detection Library | MMDetection3D 0.11.0 |
| MMCV Version | 1.2.4 |

## Reproducibility

All configuration files used for the experiments are committed to `configs/`. To reproduce a specific experiment, follow the setup in [`setup.md`](setup.md) and run:

```bash
python tools/train.py configs/<config_name>.py --work-dir work_dirs/<run_name>
```

For evaluation:

```bash
python tools/test.py configs/<config_name>.py work_dirs/<run_name>/epoch_N.pth --eval bbox
```

## Limitations

This methodology has several acknowledged limitations:

- The nuScenes mini subset (323 training samples) provides limited coverage of rare object classes (Trailers, Construction Vehicles, Motorcycles, Bicycles), which constrains the validity of conclusions about those categories.
- Environmental diversity is moderate — covering urban Boston and Singapore environments under varied lighting, but with limited extreme-weather and rural scenarios.
- All experiments use the standard nuScenes evaluation protocol; results may not directly translate to other 3D detection benchmarks (e.g., Waymo, KITTI) without re-training.
- Compute constraints required training schedules shorter than those used in the published TransFusion paper, which contributes to the absolute mAP gap observed.

These limitations are addressed honestly throughout the analysis and shape the conclusions about what the experimental evidence does and does not support.
