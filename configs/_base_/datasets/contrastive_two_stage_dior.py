# dataset settings
dataset_type = 'ContrastiveFewShotDiorDataset'
data_root = 'data/dior/'
multi_scales = (32, 64, 128, 256, 512, 800)
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
        direction=['horizontal', 'vertical', 'diagonal'])
]

train_pipeline_create_novel = [
    dict(type='AugScaled_nBBOX',
                out_max = 15,
                nbr_nBBOX = 2,
                min_bbox_size = 5,
                batch_size = batch_size, 
                nbr_class = 20,
                BBOX_scaling = 0.3
        )
]

train_pipeline_augment =[
    dict(
        type='MultiRandomFlip',
        flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='MultiRandomFlip',
        flip_ratio=.5,
        direction='vertical'),
    dict(
        type='MultiRandomFlip',
        flip_ratio=0.2,
        direction='diagonal')
]

train_pipeline_finish = [
    dict(type='Pad', size_divisor=64),
    dict(type='Normalize', **img_norm_cfg),

    dict(type='ContrastiveDefaultFormatBundle'), 

    
    dict(type='ContrastiveCollect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_labels_true'], 
                        meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg', 'applied_transformation'))
]


auxiliary=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='CropInstance', context_ratio=1 / 7.),
        dict(
            type='ResizeToMultiScale',
            multi_scales=[(s * 8 / 7., s * 8 / 7.) for s in multi_scales]),
        dict(
            type='MultiImageRandomCrop',
            multi_crop_sizes=[(s, s) for s in multi_scales]),
        dict(type='MultiImageNormalize', **img_norm_cfg),
        dict(type='MultiImageRandomFlip', flip_ratio=0.5),
        dict(type='MultiImagePad', size_divisor=32),
        dict(type='MultiImageFormatBundle'),
        dict(type='MultiImageCollect', keys=['img', 'gt_labels'])
    ]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
# classes splits are predefined in FewShotVOCDataset
multi_pipeline = dict(normal=train_pipeline_base, novel=train_pipeline_create_novel, augmented=train_pipeline_augment, finish=train_pipeline_finish, aux=auxiliary)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    auxiliary_samples_per_gpu=2,
    auxiliary_workers_per_gpu=2,
    train=dict(
        type='TwoBranchDataset',
        reweight_dataset=False,
        main_dataset=dict(
            type=dataset_type,
            ann_cfg=[
                dict(
                    type='ann_file',
                    ann_file=data_root + '/ImageSets/Main/train.txt')
            ],
            img_prefix=data_root + '/JPEGImages/',
            multi_pipelines=multi_pipeline,
            classes=None,
            use_difficult=False,
            instance_wise=False,
            dataset_name='main_dataset'),
        auxiliary_dataset=dict(
            copy_from_main_dataset=True,
            instance_wise=True,
            dataset_name='auxiliary_dataset')),
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
        test_mode=True,
        classes=None))
 