
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

angle_version = 'le90'
# HBB
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=(1024, 1024),
        keep_ratio = True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=64), 
    dict(type='FliterEmpty'),
    dict(type='DOTASpecialIgnore', ignore_diff=False, ignore_truncated=False, ignore_size=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
# classes splits are predefined in FewShotDotaDataset/HBB
# FewShotDotaDefaultDataset/HBB predefine ann_cfg for model reproducibility.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=64),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]


data_root = '/home/data/dota-1.5/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='DotaDatasetHBB',
        img_subdir=data_root + 'train/images/raw/images/',
        ann_subdir=data_root + 'train/labelTxt-v1.5/DOTA-v1.5_train_hbb/xml/xmlVOC',
        ann_file=data_root + 'train/labelTxt-v1.5/DOTA-v1.5_train_hbb/train.txt',
        pipeline=train_pipeline
        ),
    val=dict(
        type='DotaDatasetHBB',
        img_subdir=data_root + 'val/images/raw/images/',
        ann_subdir=data_root + 'val/labelTxt-v1.5/DOTA-v1.5_val_hbb/xml/xmlVOC',
        ann_file=data_root + 'val/labelTxt-v1.5/DOTA-v1.5_val_hbb/val.txt',
        pipeline=test_pipeline
    ),
    test=dict(
        type='DotaDatasetHBB',
        img_subdir=data_root + 'val/images/raw/images/',
        ann_subdir=data_root + 'val/labelTxt-v1.5/DOTA-v1.5_val_hbb/xml/xmlVOC',
        ann_file=data_root + 'val/labelTxt-v1.5/DOTA-v1.5_val_hbb/val.txt',
        pipeline=test_pipeline))
evaluation = dict(metric='mAP')