# Analysis & Evaluation Scripts

This directory contains the custom Python scripts used for evaluation, analysis, and visualization beyond the standard MMDetection3D training and testing tools.

## Script Index

### Evaluation Scripts

| Script | Purpose |
|---|---|
| `focused_map_eval.py` | Real IoU-based mAP calculation with precision-recall curves; uses 3D IoU matching to compute Average Precision against ground truth |
| `gpu_enhanced_evaluation.py` | GPU-accelerated TransFusion evaluation with weight loading from CenterPoint pre-trained checkpoints; reports timing, throughput, and high-confidence ratios |
| `calculate_real_performance.py` | Heuristic precision-recall analysis from raw detection scores; produces a quick performance estimate from a saved score tensor |

### Analysis Scripts

| Script | Purpose |
|---|---|
| `comprehensive_data_variety_analysis.py` | Analyzes nuScenes dataset variety across object classes, environmental conditions, and research-question suitability |
| `latency_optimization_analysis.py` | Computes the impact of latency optimizations (FP16, pruning, batching, early exit, simplified fusion) on inference time and accuracy |
| `generate_comprehensive_report.py` | Generates summary reports combining quantitative results, distance analysis, and per-class breakdowns |

### Inference Scripts

| Script | Purpose |
|---|---|
| `ensemble_tta_inference.py` | Test-Time Augmentation (TTA) inference combining multiple checkpoints with 8 augmentation configurations (rotations, scales, flips, combinations) |

## Usage

Most scripts are designed to run from the project root with the local TransFusion environment activated:

```bash
cd /path/to/multimodal-3d-detection-capstone
python scripts/<script_name>.py
```

Scripts assume the standard project layout with:
- `configs/` containing model configurations
- `data/nuscenes/` containing the nuScenes dataset
- `checkpoints/` containing pre-trained weights
- `work_dirs/` containing trained model outputs

## Output Locations

Scripts save outputs to:
- `results/` — JSON files, CSVs, and analysis artifacts
- `results/figures/` — Generated visualizations
- `results/reports/` — Text-based reports

## A Note on Script Quality

These scripts were developed iteratively over the course of the research and reflect the working state of the experimental investigation. Some scripts include heuristic estimations (e.g., projected mAP under different conditions) that are useful for quick analysis but are clearly distinguished from canonical measured results. The authoritative quantitative outputs of the project are in `results/quantitative_results.json` and the trained model evaluation logs.

For the full results write-up, see [`../docs/results.md`](../docs/results.md).
