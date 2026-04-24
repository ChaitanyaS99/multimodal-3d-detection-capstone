_base_ = './transfusion_nusc_voxel_LC.py'

# MAXIMUM PERFORMANCE: LiDAR-Primary with Adaptive Camera
model = dict(
    # Keep strong LiDAR backbone
    pts_backbone=dict(
        type='VoxelNet',
        # Enhanced LiDAR processing
        num_layers=4,
        layer_nums=[3, 5, 5, 5],
    ),
    
    # Lightweight camera processing
    img_backbone=dict(
        type='ResNet',
        depth=18,  # Smaller ResNet (less interference)
        num_stages=4,
        frozen_stages=2,  # Freeze early layers
    ),
    
    pts_bbox_head=dict(
        # Enhanced LiDAR focus
        loss_cls=dict(loss_weight=3.0),      # Strong classification
        loss_bbox=dict(loss_weight=1.0),     # Good localization  
        loss_heatmap=dict(loss_weight=2.0),  # Strong detection
        
        transformer=dict(
            decoder=dict(
                transformerlayers=dict(
                    # Adaptive cross-attention
                    attn_cfgs=dict(
                        cross_attn_cfg=dict(
                            embed_dims=128,
                            num_heads=4,
                            attn_drop=0.3,  # Moderate dropout
                            proj_drop=0.3,
                            dropout_layer=dict(type='Dropout', drop_prob=0.3)
                        )
                    )
                )
            )
        )
    ),
    
    # LiDAR-primary fusion weights
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=[0.1, 0.1, 0.2],
            point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        )
    ),
    
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_thr=0.2,        # Optimized NMS
            score_thr=0.05,     # Lower threshold to catch more
            min_bbox_size=0,
            nms_pre=1000,       # More candidates
            max_num=500,        # More final detections
        )
    ),
    
    # Ensure camera is helper, not leader
    use_camera=True,
    fuse_img=True,
)

# Optimized training settings
optimizer = dict(
    type='AdamW',
    lr=0.0001,           # Conservative learning rate
    weight_decay=0.001,  # Light regularization
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),  # Slow camera learning
            'pts_backbone': dict(lr_mult=1.0),  # Full LiDAR learning
        }
    )
)

# Learning schedule
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    step=[8, 12],
    gamma=0.5
)

# Training settings
runner = dict(max_epochs=15)
evaluation = dict(interval=3)
checkpoint_config = dict(interval=3)

work_dir = 'work_dirs/MAXIMUM_PERFORMANCE_LIDAR_PRIMARY'
load_from = 'work_dirs/NUCLEAR_CLEAN/epoch_20.pth'
