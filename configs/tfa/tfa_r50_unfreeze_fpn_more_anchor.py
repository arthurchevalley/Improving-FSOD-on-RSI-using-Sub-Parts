_base_ = ['../_base_/models/faster_rcnn_r50_fpn_more_anchor.py']
model = dict(
    type='TFA',
    backbone=dict(frozen_stages=4),
    
    frozen_parameters=[
        'backbone', 
        'neck'
    ],
    
    roi_head=dict(
        bbox_head=dict(
            type='CosineSimBBoxHead',
            num_shared_fcs=2,
            num_classes=20,
            cls_bias = True,
            scale=20)))
