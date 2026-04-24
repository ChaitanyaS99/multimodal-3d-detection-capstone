_base_ = './transfusion_nusc_voxel_LC.py'

# Softer training with teacher guidance
model = dict(
 pts_bbox_head=dict(
 loss_cls=dict(
 type='FocalLoss',
 use_sigmoid=True,
 gamma=1.5, # Softer than default 2.0
 alpha=0.25,
 loss_weight=1.5 # Reduced weight
 ),
 loss_bbox=dict(loss_weight=0.8),
 loss_heatmap=dict(loss_weight=1.2)
 )
)

# Gentler training schedule
optimizer = dict(
 type='AdamW',
 lr=0.00005, # Half the learning rate
 weight_decay=0.001
)

runner = dict(max_epochs=8)
evaluation = dict(interval=2)
work_dir = 'work_dirs/KNOWLEDGE_DISTILLATION'
load_from = 'work_dirs/NUCLEAR_CLEAN/epoch_20.pth'
