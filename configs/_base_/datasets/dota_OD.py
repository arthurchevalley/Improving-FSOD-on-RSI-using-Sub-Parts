
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

angle_version = 'le90'
# HBB
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=(800, 800), # 1024, 1024
        keep_ratio = True,
        bbox_clip_border = True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=64), 
    dict(type='FliterEmpty'),
    
    dict(type='DOTASpecialIgnore', ignore_diff=False, ignore_truncated=False, ignore_size=2,),
    dict(type='DefaultFormatBundle'),
    #dict(type='SaveTransformed'), 
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
# classes splits are predefined in FewShotDotaDataset/HBB
# FewShotDotaDefaultDataset/HBB predefine ann_cfg for model reproducibility.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            #dict(type='DefaultFormatBundle'),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]


data_root = '/home/data/dota-1.5/'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='FewShotDotaDatasetHBB',
        save_dataset=False,
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root + 'train/labelTxt-v1.5/DOTA-v1.5_train_hbb/train.txt')
        ],
        img_prefix=data_root + 'train/images/raw/images/',
        pipeline=train_pipeline,
        classes=None,
        min_bbox_area = 2, # added ?
        version='hbb',
        use_difficult=True,
        instance_wise=False),
    val=dict(
        type='FewShotDotaDatasetHBB',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root + 'val/labelTxt-v1.5/DOTA-v1.5_val_hbb/val.txt')
        ],
        img_prefix=data_root + 'val/images/raw/images/',
        pipeline=test_pipeline,
        version='hbb',
        use_difficult=True,
        classes=None,
    ),
    test=dict(
        type='FewShotDotaDatasetHBB',
        ann_file=data_root + 'test/',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline,
        version='hbb',
        test_mode=True))