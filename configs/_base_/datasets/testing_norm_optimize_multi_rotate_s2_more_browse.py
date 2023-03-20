
# dataset settings
dataset_type = 'ContrastiveFewShotDiorDataset'
data_root = '/home/data/dior/'
batch_size = 1

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline_base = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='FliterEmpty'),
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(
        type='RandomFlip',
        flip_ratio=0.75,
        direction=['horizontal', 'vertical', 'diagonal'])
]

train_pipeline_create_novel = [
    dict(type='AugScaled_nBBOX',
                out_max = 15,
                nbr_nBBOX = 5,
                min_bbox_size = 5,
                batch_size = batch_size, 
                nbr_class = 20,
                BBOX_scaling = 0.3
        )
]


train_pipeline_augment =[
    dict(
        type='MultiRandomFlip',
        flip_ratio=.5,
        direction='horizontal'),
    dict(
        type='MultiRandomFlip',
        flip_ratio=.5,
        direction='vertical'),
    dict(
        type='MultiRandomFlip',
        flip_ratio=.2,
        direction='diagonal'),
    dict(
        type='ContrastiveRotate',
        prob=.5),
    dict(
        type='MultiRandomFlip',
        flip_ratio=.5,
        direction='diagonal'),
    dict(
        type='ColorTransform',
        level = 10.,
        prob=.5),
    dict(
        type='ContrastTransform',
        level = 10.,
        prob=.5),
    dict(
        type='BrightnessTransform',
        level = 10.,
        prob=.5)
    
]


train_pipeline_finish = [

    dict(type='ContrastiveDefaultFormatBundle'), 

    
    dict(type='ContrastiveCollect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_labels_true'], 
                        meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg', 'applied_transformation'))
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



multi_pipeline = dict(normal=train_pipeline_base, novel=train_pipeline_create_novel, augmented=train_pipeline_augment, finish=train_pipeline_finish)
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        dior_folder_path = data_root,
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root + '/ImageSets/Main/train.txt')
            ],
        img_prefix=data_root + '/JPEGImages/',
        classes=None,
        multi_pipelines=multi_pipeline
        ),
    FTprep=dict(
        type=dataset_type,
        dior_folder_path = data_root,
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root + '/ImageSets/Main/train.txt')
            ],
        img_prefix=data_root + '/JPEGImages/',
        classes=None,
        multi_pipelines=multi_pipeline
        ),
    val=dict(
        type=dataset_type,
        dior_folder_path = data_root,
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
        dior_folder_path = data_root,
        ann_cfg=[ 
            dict(
                type='ann_file', 
                ann_file=data_root + '/ImageSets/Main/test.txt')
            ],
        img_prefix=data_root + '/JPEGImages/',
        pipeline=test_pipeline,
        classes=None))
