_base_ = [
    '../../../_base_/models/faster_rcnn_r50_bifpn.py', 
    '../../../_base_/datasets/tfa_dior.py', 
    '../../../_base_/schedules/schedule_1x.py',
    '../../../_base_/default_runtime.py'
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
        bbox_head=dict(
            num_classes=17,
            loss_bbox=dict(type='SmoothL1Loss_analyse', beta=1.0, loss_weight=1.0),
        ),
    )
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(classes='BASE_CLASSES_SPLIT3'),
    val=dict(classes='BASE_CLASSES_SPLIT3'),
    test=dict(classes='BASE_CLASSES_SPLIT3'))
evaluation = dict(
    interval=4,
    metric='mAP',
    class_splits=['BASE_CLASSES_SPLIT3'])

optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
auto_scale_lr = dict(enable=False, base_batch_size=4)
checkpoint_config = dict(interval=11)