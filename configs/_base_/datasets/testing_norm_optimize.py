
# dataset settings
dataset_type = 'FewShotDiorDataset'
data_root = 'data/dior/'
batch_size = 2

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline_base = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(
        type='RandomFlip',
        flip_ratio=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    #
    dict(type='Pad', size_divisor=64),
    dict(type='AugScaled_nBBOX',
                out_max = 15,
                nbr_nBBOX = 2,
                min_bbox_size = 5,
                batch_size = batch_size, 
                nbr_class = 19,
                BBOX_scaling = 0.3
        ),

    dict(
        type='ContrastiveRandomFlip',
        flip_ratio=0.75,
        direction='horizontal'),
    dict(
        type='ContrastiveRandomFlip',
        flip_ratio=0.75,
        direction='vertical'),
    dict(
        type='ContrastiveRandomFlip',
        flip_ratio=0.75,
        direction='diagonal'),
    dict(type='NovelNormalize', **img_norm_cfg),

    #dict(type='DefaultFormatBundle'), 
    dict(type='ContrastiveDefaultFormatBundle', to_format = ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']),
    
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
    samples_per_gpu=batch_size,
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
