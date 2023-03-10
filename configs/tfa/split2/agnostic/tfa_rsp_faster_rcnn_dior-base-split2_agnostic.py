_base_ = [
    'faster_rcnn_r50_fpn_agnostic.py', 
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
            with_cls=False,
            reg_class_agnostic = True,
            num_classes=15,
            loss_bbox=dict(type='SmoothL1Loss_analyse', beta=1.0, loss_weight=1.0),
        ),
    )
)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
       # dict(type='CometMLLoggerHook', 
        #    project_name='logger_comet_ml',
         #   api_key= 'UavGjAWatUgY4kp6T3tv3VWuS')
    ])

data = dict(
    train=dict(classes='BASE_CLASSES_SPLIT2'),
    val=dict(classes='BASE_CLASSES_SPLIT2'),
    test=dict(classes='BASE_CLASSES_SPLIT2'))
evaluation = dict(
    interval=1,
    metric='mIoU',
    class_splits=['BASE_CLASSES_SPLIT2'])

optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
auto_scale_lr = dict(enable=False, base_batch_size=2)
checkpoint_config = dict(interval=11)