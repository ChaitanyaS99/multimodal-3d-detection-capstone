# Nuclear Power Configuration
_base_ = ['./transfusion_standalone.py']

# NUCLEAR DATA CONFIG
data = dict(
    samples_per_gpu=8,  # Start with 8 to avoid OOM
    workers_per_gpu=8,
    persistent_workers=True,
)

# NUCLEAR OPTIMIZATION  
optimizer = dict(
    type='AdamW', 
    lr=0.001,  # Higher LR for multi-GPU
    weight_decay=0.01
)

# Enhanced loss weights for better performance
model = dict(
    pts_bbox_head=dict(
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0  # 2x weight
        ),
        loss_heatmap=dict(
            type='GaussianFocalLoss', 
            loss_weight=2.0  # 2x weight
        )
    )
)

# Extended training
total_epochs = 20
runner = dict(type='EpochBasedRunner', max_epochs=20)

# Advanced LR schedule
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    min_lr_ratio=0.01
)

# Continue from your best checkpoint
load_from = './work_dirs/REAL_TRAINING_FINAL/epoch_3.pth'
work_dir = './work_dirs/NUCLEAR_POWER'
