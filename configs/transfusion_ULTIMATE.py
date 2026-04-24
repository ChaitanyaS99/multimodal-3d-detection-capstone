_base_ = './transfusion_nusc_voxel_LC.py'

# CRITICAL FIX 1: FocalLoss for severe class imbalance
model = dict(
    pts_bbox_head=dict(
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=4.0  # Higher weight for classification
        ),
        loss_bbox=dict(loss_weight=0.25),
        loss_heatmap=dict(loss_weight=1.0)
    )
)

# CRITICAL FIX 2: Much more conservative learning rate  
optimizer = dict(
    type='AdamW',
    lr=0.00005,  # 5x lower than your current rate
    betas=(0.9, 0.999),
    weight_decay=0.01
)

# CRITICAL FIX 3: Stable step-based schedule (not cyclic)
lr_config = dict(
    policy='step',
    step=[20, 28],  
    gamma=0.1,
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.1
)

# Remove problematic momentum config
momentum_config = None

# CRITICAL FIX 4: Extended training for convergence
runner = dict(type='EpochBasedRunner', max_epochs=35)

# CRITICAL FIX 5: Frequent validation to track progress
evaluation = dict(interval=2, save_best='pts_bbox_NuScenes/mAP')

# Better checkpoint saving
checkpoint_config = dict(interval=5, save_optimizer=False)

# Clean work directory
work_dir = 'work_dirs/transfusion_ULTIMATE_FIXED'

# Use your proven starting checkpoint
load_from = 'work_dirs/NUCLEAR_CLEAN/epoch_20.pth'
