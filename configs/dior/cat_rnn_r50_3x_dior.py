_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py', 
    '../_base_/datasets/dior_OD.py', 
    '../_base_/schedules/schedule_3x.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        depth=50
    ),
    neck = dict(
        type='DenseFPNSCP',
        stack_times=5,
        norm_cfg=dict(type='BN', requires_grad=True)
        #_insert_=True, 
        
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
            num_classes=20,
            #loss_bbox=dict(type='SmoothL1Loss_analyse', beta=1.0, loss_weight=1.0),
    #        reg_class_agnostic = True,
        ),
    )
)

auto_scale_lr = dict(enable=False, base_batch_size=1)
