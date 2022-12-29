_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/dota_OD_clean.py',
    '../_base_/schedules/schedule_3x.py', 
    '../_base_/default_runtime.py'
]

    
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet101',
        ),
        dcn=dict(
            type='DCN',
            deform_groups=1, 
            fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True
        )
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=16,
        ),
    ),
    train_cfg=dict(
        rcnn=dict(
            sampler=dict(
                type='OHEMSampler'
            )
        )
    ),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='soft_nms', iou_threshold=0.3),
            max_per_img=100
        )
    )
        
    )

