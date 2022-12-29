_base_ = [
    '../../_base_/models/contrastive_r50_noRoI.py', 
    '../../_base_/datasets/testing_norm_optimize_multi_rotate.py', 
    '../../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py'
]


# model settings

model = dict(
    
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='rsp-resnet-50-ckpt_ready.pth',
            )
    ),
    rpn_head=dict(
            loss_bbox=dict(type='SmoothL1Loss_analyse', beta = 1.0/9.0,loss_weight=1.0)
    ),
    roi_head=dict(
        type='DualCosContRoIHead_Branch',
        bbox_head=dict(
            type='QueueAugContrastiveBBoxHead_Branch',
            mlp_head_channels=128,
            with_weight_decay=True,
            loss_bbox=dict(type='SmoothL1Loss_analyse', beta=1.0, loss_weight=1.0),
            loss_cosine=dict(
                type='QueueDualSupervisedContrastiveLoss',
                temperature=0.2,
                iou_threshold=0.8,
                loss_weight=0.25,
                reweight_type='none'),
            loss_c_cls=dict(
                type='QueueDualSupervisedContrastiveLoss',
                temperature=0.2,
                iou_threshold=0.8,
                loss_weight=0.25,
                reweight_type='none'),
            loss_base_aug=dict(
                type='QueueDualSupervisedContrastiveLoss',
                temperature=0.2,
                iou_threshold=0.8,
                loss_weight=0.25,
                reweight_type='none'),
            loss_c_bbox=dict(
                type='IoULossBBOX',
                loss_weight=0.25),
            queue_path = 'init_queue.p',
            use_queue = False,
            to_norm_cls = True,
            main_training = True,
            num_classes=17,
            scale=20,
            init_cfg=[
                dict(
                    type='Caffe2Xavier',
                    override=dict(type='Caffe2Xavier', name='shared_fcs')),
                dict(
                    type='Normal',
                    override=dict(type='Normal', name='fc_cls', std=0.01)),
                dict(
                    type='Normal',
                    override=dict(type='Normal', name='fc_reg', std=0.001)),
                dict(
                    type='Caffe2Xavier',
                    override=dict(
                        type='Caffe2Xavier', name='contrastive_head'))
            ])),

)


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(classes='BASE_CLASSES_SPLIT3'),
    val=dict(classes='BASE_CLASSES_SPLIT3'),
    test=dict(classes='BASE_CLASSES_SPLIT3'))
evaluation = dict(
    interval=4,
    metric='mAP',
    class_splits=['BASE_CLASSES_SPLIT3'])
auto_scale_lr = dict(enable=False, base_batch_size=2)
checkpoint_config = dict(interval=4)
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11, 16])
runner = dict(type='EpochBasedRunner', max_epochs=18)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='CometMLLoggerHook', 
            project_name='logger_comet_ml',
            api_key= 'UavGjAWatUgY4kp6T3tv3VWuS')
    ])