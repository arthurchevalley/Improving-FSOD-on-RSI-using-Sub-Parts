_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn.py', 
    '../../_base_/datasets/tfa_dior.py', 
    '../../_base_/default_runtime.py'
]


# classes splits are predefined in FewShotVOCDataset
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(classes='BASE_CLASSES_SPLIT3'),
    val=dict(classes='BASE_CLASSES_SPLIT3'),
    test=dict(classes='BASE_CLASSES_SPLIT3'))

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
# using regular sampler can get a better base model
use_infinite_sampler = False


optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[60000, 80000])
runner = dict(type='IterBasedRunner', max_iters=90000)

evaluation = dict(
    interval=30000,
    metric='mAP',
    class_splits=['BASE_CLASSES_SPLIT3'])

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
auto_scale_lr = dict(enable=False, base_batch_size=4)
checkpoint_config = dict(interval=30000)