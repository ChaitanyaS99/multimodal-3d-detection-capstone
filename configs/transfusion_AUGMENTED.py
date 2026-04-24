_base_ = './transfusion_nusc_voxel_LC.py'

# Freeze backbones, aggressive augmentation
model = dict(
    img_backbone=dict(frozen_stages=3),
    pts_backbone=dict(frozen_stages=2)
)

# 5x more augmentation to artificially increase dataset
train_pipeline = [
    # Aggressive point cloud augmentation
    dict(type='GlobalRotScaleTrans', 
         rot_range=[-1.57, 1.57],      # Full rotation
         scale_ratio_range=[0.7, 1.3], # Heavy scaling
         translation_std=[2, 2, 2]),   # More translation
    
    # Aggressive image augmentation  
    dict(type='MultiViewWrapper', transforms=[
        dict(type='PhotoMetricDistortion3D',
             brightness_delta=64,
             contrast_range=(0.2, 1.8),
             saturation_range=(0.2, 1.8)),
        dict(type='RandomFlip', flip_ratio=0.5),
    ]),
    # ... rest of pipeline
]

optimizer = dict(lr=0.00005)
runner = dict(max_epochs=80)  # More epochs with augmentation

work_dir = 'work_dirs/transfusion_HEAVY_AUGMENTED'
