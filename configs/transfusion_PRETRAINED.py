_base_ = './transfusion_nusc_voxel_LC.py'

model = dict(
 img_backbone=dict(
 frozen_stages=2, # Freeze early layers
 norm_eval=True # Use pre-trained features
 ),
 pts_backbone=dict(
 frozen_stages=1 # Freeze some LiDAR layers too
 )
)

# Much lower learning rate for pre-trained features
optimizer = dict(lr=0.00001)

work_dir = 'work_dirs/transfusion_PRETRAINED'
