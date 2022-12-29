_base_ = [
    '../../_base_/models/faster_rcnn_CIR_r50.py', 
    '../../_base_/datasets/tfa_dior.py', 
    '../../_base_/schedules/schedule_1x.py',
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
)

data = dict(
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
custom_hooks = [dict(type='NumClassCheckHook')
    ]
# first ch. = 2 et nbr = 2
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
auto_scale_lr = dict(enable=False, base_batch_size=2)
checkpoint_config = dict(interval=11)

