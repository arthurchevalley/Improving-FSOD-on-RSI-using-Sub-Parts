_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py', 
    '../_base_/datasets/dota_OD_clean.py', 
    '../_base_/schedules/schedule_3x.py',
    '../_base_/default_runtime.py'
]

# model settings

model = dict(
    #rpn_head=dict(
    #        loss_bbox=dict(type='SmoothL1Loss_analyse', beta = 1.0/9.0,loss_weight=1.0)
    #),
    roi_head=dict(
        bbox_head=dict(
            num_classes=16,
    #        loss_bbox=dict(type='SmoothL1Loss_analyse', beta=1.0, loss_weight=1.0),
        ),
    )
)
auto_scale_lr = dict(enable=False, base_batch_size=2)
