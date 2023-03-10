_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn.py', 
    '../../_base_/datasets/tfa_dior.py', 
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
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8, 16],# 8, 16, 32
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
            loss_bbox=dict(type='SmoothL1Loss_analyse', beta = 1.0/9.0,loss_weight=1.0)
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=15,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss_analyse', beta=1.0, loss_weight=1.0),
        ),
    )
)

data = dict(
    train=dict(classes='BASE_CLASSES_SPLIT2'),
    val=dict(classes='BASE_CLASSES_SPLIT2'),
    test=dict(classes='BASE_CLASSES_SPLIT2'))
evaluation = dict(
    interval=4,
    metric='mAP',
    class_splits=['BASE_CLASSES_SPLIT2'])

optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
auto_scale_lr = dict(enable=False, base_batch_size=2)
checkpoint_config = dict(interval=11)