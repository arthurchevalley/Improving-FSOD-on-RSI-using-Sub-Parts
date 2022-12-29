
# dataset settings
dataset_type = 'FewShotDiorDataset'
data_root = 'data/dior/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(
        type='RandomFlip',
        flip_ratio=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=64),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
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
            dict(type='Pad', size_divisor=64),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]


        
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root + '/ImageSets/Main/train.txt')
            ],
        img_prefix=data_root + '/JPEGImages/',
        pipeline=train_pipeline,
        classes=None),
    val=dict(
        type=dataset_type,
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root + '/ImageSets/Main/val.txt')
            ],
        img_prefix=data_root + '/JPEGImages/',
        pipeline=test_pipeline,
        classes=None),
    test=dict(
        type=dataset_type,
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root + '/ImageSets/Main/test.txt')
            ],
        img_prefix=data_root + '/JPEGImages/',
        pipeline=test_pipeline,
        classes=None))
