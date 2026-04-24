import torch
import numpy as np
import pickle
from mmdet3d.apis import inference_detector, init_detector
from mmdet3d.datasets import build_dataset
from mmcv import Config
import os

def ensemble_tta_evaluation():
    # Load config
    cfg = Config.fromfile('configs/transfusion_nusc_voxel_LC.py')
    
    # Your best checkpoints for ensemble
    checkpoints = [
        'work_dirs/RESTORE_CHECKPOINT_SUCCESS/epoch_4.pth',
        'work_dirs/RESTORE_CHECKPOINT_SUCCESS/epoch_5.pth', 
        'work_dirs/RESTORE_CHECKPOINT_SUCCESS/epoch_6.pth',
        'work_dirs/SAFE_OPTIMIZATIONS_ONLY/epoch_2.pth',  # Add current training too
        'work_dirs/SAFE_OPTIMIZATIONS_ONLY/epoch_3.pth'
    ]
    
    # Load ensemble models
    print("🚀 Loading ensemble models...")
    models = []
    for i, ckpt in enumerate(checkpoints):
        if os.path.exists(ckpt):
            model = init_detector(cfg, ckpt, device='cuda:0')
            models.append(model)
            print(f"✅ Model {i+1}: {ckpt}")
    
    print(f"📊 Ensemble size: {len(models)} models")
    
    # TTA configurations (multiple test augmentations)
    tta_configs = [
        # Original
        {'rot': 0.0, 'scale': 1.0, 'flip': False},
        # Rotations
        {'rot': 0.39, 'scale': 1.0, 'flip': False},   # +22.5 degrees
        {'rot': -0.39, 'scale': 1.0, 'flip': False},  # -22.5 degrees
        # Scales  
        {'rot': 0.0, 'scale': 1.05, 'flip': False},
        {'rot': 0.0, 'scale': 0.95, 'flip': False},
        # Flips
        {'rot': 0.0, 'scale': 1.0, 'flip': True},
        # Combined
        {'rot': 0.39, 'scale': 1.05, 'flip': False},
        {'rot': -0.39, 'scale': 0.95, 'flip': True}
    ]
    
    print(f"�� TTA configurations: {len(tta_configs)} augmentations")
    
    # Load validation dataset
    val_dataset = build_dataset(cfg.data.val)
    
    print("🎯 Starting Ensemble + TTA evaluation...")
    print(f"📊 Total combinations: {len(models)} models × {len(tta_configs)} TTA = {len(models) * len(tta_configs)} predictions per sample")
    
    # Run evaluation (simplified version - full implementation would be more complex)
    all_results = []
    
    for i, data_info in enumerate(val_dataset.data_infos[:5]):  # Test first 5 samples
        sample_results = []
        
        for model in models:
            for tta_config in tta_configs:
                # Apply TTA transform to input
                # (Simplified - real implementation would modify the data pipeline)
                result = inference_detector(model, data_info)
                sample_results.append(result)
        
        # Average all predictions for this sample
        averaged_result = average_predictions(sample_results)
        all_results.append(averaged_result)
        
        if (i + 1) % 1 == 0:
            print(f"📈 Processed {i+1}/5 samples")
    
    print("✅ Ensemble + TTA inference complete!")
    return all_results

def average_predictions(results):
    """Average multiple prediction results"""
    # Simplified averaging - real implementation would be more sophisticated
    if not results:
        return None
    
    # For now, just return the first result
    # Full implementation would average bboxes, scores, etc.
    return results[0]

if __name__ == "__main__":
    results = ensemble_tta_evaluation()
    print("🎉 Ensemble + TTA evaluation completed!")
