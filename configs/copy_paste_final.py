point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
voxel_size = [0.075, 0.075, 0.2]

# AGGRESSIVE AUGMENTATION PIPELINE
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=[0, 1, 2, 3, 4]),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=10, use_dim=[0, 1, 2, 3, 4]),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='GlobalRotScaleTrans', 
         rot_range=[-1.57079632679, 1.57079632679],
         scale_ratio_range=[0.7, 1.4],
         translation_std=[1.5, 1.5, 0.3]),
    dict(type='RandomFlip3D', 
         flip_ratio_bev_horizontal=0.8,
         flip_ratio_bev_vertical=0.8),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=[0, 1, 2, 3, 4]),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=10, use_dim=[0, 1, 2, 3, 4]),
    dict(type='MultiScaleFlipAug3D', img_scale=(1333, 800), pts_scale_ratio=1, flip=False,
         transforms=[
            dict(type='GlobalRotScaleTrans', rot_range=[0, 0], scale_ratio_range=[1.0, 1.0]),
            dict(type='RandomFlip3D'),
            dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
            dict(type='Collect3D', keys=['points'])
         ])
]

data = dict(
    samples_per_gpu=6,
    workers_per_gpu=8,
    persistent_workers=True,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type='NuScenesDataset',
            data_root='data/nuscenes/',
            ann_file='data/nuscenes/nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            test_mode=False,
            box_type_3d='LiDAR'
        )
    ),
    val=dict(
        type='NuScenesDataset',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'
    ),
    test=dict(
        type='NuScenesDataset',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'
    )
)

model = dict(
    type='TransFusionDetector',
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
    pts_bbox_head=dict(
        type='TransFusionHead',
        num_proposals=400,
        auxiliary=True,
        in_channels=512,
        hidden_channel=128,
        num_classes=len(class_names),
        num_decoder_layers=1,
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
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=6.0),
        loss_bbox=dict(type='L1Loss', loss_weight=2.0),
        loss_heatmap=dict(type='GaussianFocalLoss', loss_weight=6.0)
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

optimizer = dict(
    type='AdamW', 
    lr=0.002,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pts_bbox_head': dict(lr_mult=0.5),
            'pts_backbone': dict(lr_mult=1.5),
        }
    )
)

optimizer_config = dict(grad_clip=dict(max_norm=1.0, norm_type=2))

# STANDARD COSINE ANNEALING (compatible)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.05,
    min_lr_ratio=0.0001
)

total_epochs = 40
runner = dict(type='EpochBasedRunner', max_epochs=40)
checkpoint_config = dict(interval=5, max_keep_ckpts=8)
log_config = dict(
    interval=25,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)

evaluation = dict(
    interval=5,
    pipeline=test_pipeline,
    metric=['bbox'],
    save_best='NDS',
    rule='greater'
)

dist_params = dict(backend='nccl', init_method='env://')
log_level = 'INFO'
work_dir = './work_dirs/COPY_PASTE_FINAL'
load_from = './work_dirs/NUCLEAR_CLEAN/epoch_20.pth'
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = False
fp16 = dict(loss_scale='dynamic')
auto_scale_lr = dict(enable=True, base_batch_size=36)
seed = 42
deterministic = False
cudnn_benchmark = True
