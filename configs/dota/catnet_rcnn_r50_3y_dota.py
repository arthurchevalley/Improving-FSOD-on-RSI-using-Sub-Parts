_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py', 
    #'../_base_/modules/dfpn.py',
    #'../_base_/modules/scp.py', 
    #'../_base_/modules/hroie.py',
    '../_base_/datasets/dota_OD.py', 
    '../_base_/schedules/schedule_3x.py',
    '../_base_/default_runtime.py'
]

# model settings

model = dict(
    backbone=dict(
        depth=50
    ),
    neck=dict(
        type='DenseFPN',
        stack_times=5,
        norm_cfg=dict(type='BN', requires_grad=True),
        ),
    #neck=dict(
    #    type='DenseFPN',
    #    stack_times=5,
    #    norm_cfg=dict(type='SyncBN', requires_grad=True),
    #    add_extra_convs=False,
        
    #    _insert_=True, 
    #    type='SCP', 
    #    in_channels=256, 
    #    num_levels=5
    #),

    rpn_head=dict(
            loss_bbox=dict(type='SmoothL1Loss_analyse', beta = 1.0/9.0,loss_weight=1.0)
    ),
    roi_head=dict(
        bbox_roi_extractor=dict(
            #_refine_=True, 
            type='HRoIE', 
            direction='bottom_up'),
        mask_roi_extractor=dict(
            #_refine_=True, 
            type='HRoIE', 
            direction='top_down'),
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