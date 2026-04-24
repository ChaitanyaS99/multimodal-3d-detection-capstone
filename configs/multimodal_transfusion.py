# configs/multimodal_transfusion.py
# Full LiDAR + Camera TransFusion - Expected 60-65% mAP

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
class_names = [
 'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
voxel_size = [0.075, 0.075, 0.2]

# Multi-modal train pipeline (LiDAR + Images)
train_pipeline = [
 dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=[0, 1, 2, 3, 4]),
 dict(type='LoadPointsFromMultiSweeps', sweeps_num=10, use_dim=[0, 1, 2, 3, 4]),
 dict(type='LoadMultiViewImageFromFiles'),
 dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
 
 dict(type='GlobalRotScaleTrans', 
 rot_range=[-0.78539816, 0.78539816],
 scale_ratio_range=[0.9, 1.1],
 translation_std=[0.5, 0.5, 0.1]),
 dict(type='RandomFlip3D', 
 flip_ratio_bev_horizontal=0.5,
 flip_ratio_bev_vertical=0.5),
 
 dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
 dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
 dict(type='PointShuffle'),
 
 dict(type='DefaultFormatBundle3D', class_names=class_names),
 dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'img'])
]

# Test pipeline
test_pipeline = [
 dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=[0, 1, 2, 3, 4]),
 dict(type='LoadPointsFromMultiSweeps', sweeps_num=10, use_dim=[0, 1, 2, 3, 4]),
 dict(type='LoadMultiViewImageFromFiles'),
 dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
 dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
 dict(type='Collect3D', keys=['points', 'img'])
]

# Data config
data = dict(
 samples_per_gpu=1, # Start with 1 to avoid memory issues
 workers_per_gpu=4,
 train=dict(
 type='NuScenesDataset',
 data_root='data/nuscenes/',
 ann_file='data/nuscenes/nuscenes_infos_train.pkl',
 pipeline=train_pipeline,
 classes=class_names,
 modality=dict(use_lidar=True, use_camera=True),
 test_mode=False,
 box_type_3d='LiDAR'
 ),
 val=dict(
 type='NuScenesDataset',
 data_root='data/nuscenes/',
 ann_file='data/nuscenes/nuscenes_infos_val.pkl',
 pipeline=test_pipeline,
 classes=class_names,
 modality=dict(use_lidar=True, use_camera=True),
 test_mode=True,
 box_type_3d='LiDAR'
 )
)

# Use existing base model structure with multi-modal extensions
model = dict(
 type='TransFusionDetector',
 
 # Add image backbone
 img_backbone=dict(
 type='ResNet',
 depth=50,
 num_stages=4,
 out_indices=(0, 1, 2, 3),
 frozen_stages=1,
 norm_cfg=dict(type='BN2d', requires_grad=True),
 norm_eval=True,
 style='pytorch',
 
 ),
 
 # Add image neck
 img_neck=dict(
 type='FPN',
 in_channels=[256, 512, 1024, 2048],
 out_channels=256,
 start_level=1,
 add_extra_convs='on_output',
 num_outs=5,
 norm_cfg=dict(type='BN2d', requires_grad=True),
 relu_before_extra_convs=True
 ),
 
 # Keep your existing LiDAR components
 pts_voxel_layer=dict(
 max_num_points=15,
 voxel_size=voxel_size,
 max_voxels=(180000, 220000),
 point_cloud_range=point_cloud_range
 ),
 pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
 pts_middle_encoder=dict(
 type='SparseEncoder',
 in_channels=5,
 sparse_shape=[41, 1440, 1440],
 output_channels=128,
 order=('conv', 'norm', 'act'),
 encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
 encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
 block_type='basicblock'
 ),
 pts_backbone=dict(
 type='SECOND',
 in_channels=256,
 out_channels=[128, 256],
 layer_nums=[5, 5],
 layer_strides=[1, 2],
 norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
 conv_cfg=dict(type='Conv2d', bias=False)
 ),
 pts_neck=dict(
 type='SECONDFPN',
 in_channels=[128, 256],
 out_channels=[256, 256],
 upsample_strides=[1, 2],
 norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
 upsample_cfg=dict(type='deconv', bias=False),
 use_conv_for_no_stride=True
 ),
 
 # Enhanced fusion head
 pts_bbox_head=dict(
 type='TransFusionHead',
 num_proposals=200,
 auxiliary=True,
 in_channels=512,
 hidden_channel=128,
 num_classes=len(class_names),
 num_decoder_layers=2, # Two layers for fusion
 num_heads=8,
 learnable_query_pos=False,
 initialize_by_heatmap=True,
 nms_kernel_size=3,
 ffn_channel=256,
 dropout=0.1,
 bn_momentum=0.1,
 activation='relu',
 common_heads=dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
 bbox_coder=dict(
 type='TransFusionBBoxCoder',
 pc_range=[-54.0, -54.0],
 voxel_size=[0.075, 0.075],
 out_size_factor=8,
 post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
 score_threshold=0.0,
 code_size=10
 ),
 loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
 loss_bbox=dict(type='L1Loss', loss_weight=0.25),
 loss_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0)
 ),
 train_cfg=dict(
 pts=dict(
 dataset='nuScenes',
 assigner=dict(
 type='HungarianAssigner3D',
 iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
 cls_cost=dict(type='FocalLossCost', gamma=2.0, alpha=0.25, weight=0.15),
 reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
 iou_cost=dict(type='IoU3DCost', weight=0.25)
 ),
 pos_weight=-1,
 gaussian_overlap=0.1,
 min_radius=2,
 grid_size=[1440, 1440, 40],
 voxel_size=voxel_size,
 out_size_factor=8,
 code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
 point_cloud_range=point_cloud_range
 )
 ),
 test_cfg=dict(
 pts=dict(
 dataset='nuScenes',
 grid_size=[1440, 1440, 40],
 out_size_factor=8,
 pc_range=[-54.0, -54.0],
 voxel_size=[0.075, 0.075],
 nms_type=None
 )
 )
)

# Training setup
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(
 policy='CosineAnnealing',
 warmup='linear',
 warmup_iters=500,
 warmup_ratio=0.001,
 min_lr_ratio=0.001
)

total_epochs = 26
runner = dict(type='EpochBasedRunner', max_epochs=26)
checkpoint_config = dict(interval=5, max_keep_ckpts=3)
log_config = dict(
 interval=50,
 hooks=[
 dict(type='TextLoggerHook'),
 dict(type='TensorboardLoggerHook')
 ]
)

evaluation = dict(interval=5, pipeline=test_pipeline)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/MULTIMODAL_TRANSFUSION'
load_from = './work_dirs/SINGLE_GPU_AGGRESSIVE/epoch_10.pth'
resume_from = None
workflow = [('train', 1)]

find_unused_parameters = False
fp16 = dict(loss_scale='dynamic')
seed = 42
deterministic = False
cudnn_benchmark = True
