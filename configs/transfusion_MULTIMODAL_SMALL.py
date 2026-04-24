_base_ = './transfusion_nusc_voxel_LC.py'

# CRITICAL: Freeze most of the network, only train fusion layers
model = dict(
 img_backbone=dict(
 frozen_stages=4, # Freeze ALL image backbone 
 norm_eval=True
 ),
 pts_backbone=dict(
 frozen_stages=3 # Freeze most LiDAR backbone
 ),
 pts_bbox_head=dict(
 # Reduce complexity drastically
 num_decoder_layers=2, # Minimal layers
 num_heads=2, # Minimal attention
 loss_cls=dict(
 type='FocalLoss',
 use_sigmoid=True,
 gamma=2.0,
 alpha=0.25,
 loss_weight=8.0 # Force focus on classification
 )
 )
)

# VERY conservative learning for small dataset
optimizer = dict(
 type='AdamW',
 lr=0.00002, # Very slow learning
 weight_decay=0.01
)

# Simple step schedule
lr_config = dict(
 policy='step',
 step=[30, 45],
 gamma=0.5
)

# More epochs for small dataset
runner = dict(max_epochs=60)

# Frequent validation
evaluation = dict(interval=3)

work_dir = 'work_dirs/transfusion_MULTIMODAL_SMALL'
