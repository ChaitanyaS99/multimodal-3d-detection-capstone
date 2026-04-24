# Multi-Modal 3D Object Detection for Autonomous Driving

A research capstone investigating multi-modal 3D object detection using the **TransFusion** architecture on the **nuScenes** benchmark, with a systematic evaluation of optimization techniques across the detection pipeline.

**Author:** Chaitanya Suralkar
**Advisor:** Dr. Eungjoo Lee — University of Arizona
**Program:** M.S. Data Science, University of Arizona
**Course:** INFO 698 — Capstone (Spring 2026)

---

## Project Overview

Autonomous driving perception requires accurate 3D object detection from multiple sensor modalities. LiDAR provides precise geometric structure, while cameras contribute rich semantic context. Effectively combining these modalities remains an active research challenge due to differences in representation, alignment, and learning dynamics.

This project implements a **TransFusion-based multimodal 3D detection pipeline** on the nuScenes dataset and conducts a structured experimental study to analyze model behavior across fusion strategies, training configurations, and optimization techniques. The work establishes a reproducible baseline, evaluates a comprehensive suite of optimization approaches, and identifies promising directions for improving robustness through physics-informed methods.

## Key Highlights

- Complete reproduction and validation of the TransFusion architecture (8.3M parameters, matching the original paper)
- Systematic evaluation of **40+ configuration variants** spanning loss rebalancing, knowledge distillation, FP16 quantization, data augmentation, threshold tuning, and model ensembling
- Strong baseline performance on safety-critical classes (Cars **37.4% AP**, Pedestrians **43.2% AP**)
- Detailed distance-threshold analysis revealing range-aware model behavior (Cars **56.4% AP** at 4m, Pedestrians **52.7% AP** at 4m)
- Empirical scaling-law fit (mAP ∝ N^0.52) projecting full-dataset performance consistent with the published TransFusion benchmark of 65.5% mAP
- Comprehensive data variety assessment characterizing the strengths and limitations of the nuScenes mini subset for research

## Repository Structure

```
multimodal-3d-detection-capstone/
│
├── README.md                          Project overview and entry point
├── requirements.txt                   Python dependencies
│
├── docs/                              Detailed documentation
│   ├── methodology.md                 Approach and experimental design
│   ├── experiments.md                 Full experimental suite documented
│   ├── results.md                     Findings, figures, and analysis
│   ├── architecture.md                Model architecture details
│   └── setup.md                       Installation and data preparation
│
├── configs/                           Experimental configurations
│   ├── README.md                      Guide to all config files
│   └── transfusion_*.py               TransFusion variants and experiments
│
├── scripts/                           Analysis and evaluation scripts
│   ├── README.md                      Guide to all scripts
│   └── *.py                           Evaluation, analysis, and inference tools
│
├── results/                           Findings and artifacts
│   ├── quantitative_results.json      Per-class AP and detection statistics
│   ├── class_performance_data.csv     Tabular per-class results
│   ├── figures/                       Generated charts and visualizations
│   └── reports/                       Technical reports and assessments
│
└── poster/                            Capstone poster
    └── Suralkar_Capstone_Poster.pdf
```

## Getting Started

For installation and dataset preparation, see [`docs/setup.md`](docs/setup.md).

For a detailed walkthrough of the methodology and experimental design, see [`docs/methodology.md`](docs/methodology.md).

For the full experimental suite and what each configuration tests, see [`docs/experiments.md`](docs/experiments.md).

For results and analysis, see [`docs/results.md`](docs/results.md).

## Research Direction

Building on the findings of this work, the next research direction explores **physics-informed query initialization** — embedding ground-plane constraints, dimension-aware object priors, and geometric reasoning directly into the model's query generation stage. The goal is to move multimodal perception toward geometry-aware, physically grounded systems that are more robust under sparse data, occlusions, and sensor inconsistencies.

## Acknowledgements

This work builds on the official TransFusion implementation by Bai et al. (CVPR 2022) and uses the **nuScenes** dataset by nuTonomy. Special thanks to **Dr. Eungjoo Lee** at the University of Arizona for research supervision and guidance throughout the project, and to **Dr. Nitika Sharma** for capstone instruction.

## References

- Bai, X., Hu, Z., Zhu, X., Huang, Q., Chen, Y., Fu, H., & Tai, C. L. (2022). *TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers*. CVPR 2022. [arXiv:2203.11496](https://arxiv.org/abs/2203.11496)
- Caesar, H., et al. (2020). *nuScenes: A Multimodal Dataset for Autonomous Driving*. CVPR 2020. [arXiv:1903.11027](https://arxiv.org/abs/1903.11027)

---

*© 2026 Chaitanya Suralkar. All rights reserved.*
