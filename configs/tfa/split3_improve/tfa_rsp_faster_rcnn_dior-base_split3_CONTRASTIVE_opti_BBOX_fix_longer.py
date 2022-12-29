_base_ = [
    '../../_base_/models/CRFCNN_r50_opti.py', 
    '../../_base_/datasets/testing_norm_optimize_multi.py', 
    '../../_base_/schedules/schedule_1x_nwu.py',
    '../../_base_/default_runtime.py'
]


# model settings
number_of_class = 17
number_of_contrastive_class = 3
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
        bbox_head=dict(
            num_classes=number_of_class,
            loss_bbox=dict(type='SmoothL1Loss_analyse', beta=1.0, loss_weight=1.0),
        ),
    ),
    c_rpn_head=dict(
            loss_bbox=dict(type='SmoothL1Loss_analyse', beta = 1.0/9.0,loss_weight=1.0)
    ),
    c_roi_head=dict(
        bbox_head=dict(
            num_classes=number_of_class,
            loss_cls=dict(
                type='ConstellationLossCLS', 
                K=2, 
                max_contrastive_loss = 2, 
                loss_weight=.2
            ),
            loss_bbox=dict(
                type='ConstellationLossBBOX', 
                beta = 1.0/9.0,
                K=2, 
                max_contrastive_loss = 2, 
                loss_weight=.01
            )
        ),
    )
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        classes='BASE_CLASSES_SPLIT3'),
    val=dict(
            classes='BASE_CLASSES_SPLIT3'),
    test=dict(
            classes='BASE_CLASSES_SPLIT3'))
evaluation = dict(
    interval=4,
    metric='mAP',
    class_splits=['BASE_CLASSES_SPLIT3'])
custom_hooks = [dict(type='NumClassCheckHook'),
                dict(type='ContrastiveLossWeight',
                change_weight_epoch_cls=1, # 3
                change_weight_epoch_bbox=1, # 2
                increase_step_cls = .2,
                increase_step_bbox = .05, # 0.05 
                number_of_epoch_step_bbox = 1, # 2
                number_of_epoch_step_cls = 6) # 
                ]
# first ch. = 2 et nbr = 2
optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0001)
lr_config = dict(
    policy='step',
    warmup_ratio=0.001,
    step=[8])


auto_scale_lr = dict(enable=False, base_batch_size=2)
checkpoint_config = dict(interval=2)

load_from = ('work_dirs/tfa_rsp_faster_rcnn_dior-base_split3_CONTRASTIVE_opti_BBOX/true_split3_extended.pth')

