_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py', 
    '../_base_/datasets/dota_OD.py', 
    '../_base_/schedules/schedule_3x.py',
    '../_base_/default_runtime.py'
]

# model settings

model = dict(
    rpn_head=dict(
            loss_bbox=dict(type='SmoothL1Loss_analyse', beta = 1.0/9.0,loss_weight=1.0)
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=15,
            loss_bbox=dict(type='SmoothL1Loss_analyse', beta=1.0, loss_weight=1.0),
    #        reg_class_agnostic = True,
        ),
    )
)
data = dict(
    train=dict(classes='BASE_CLASSES_SPLIT1'),
    val=dict(classes='BASE_CLASSES_SPLIT1'),
    test=dict(classes='BASE_CLASSES_SPLIT1'))
evaluation = dict(
    interval=4,
    metric='mAP',
    class_splits=['BASE_CLASSES_SPLIT1'])


# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    step=[2]
    )

runner = dict(type='EpochBasedRunner', max_epochs=4)

load_from = 'work_dirs/cat_rcnn_r50_3y_dota/epoch_32.pth'