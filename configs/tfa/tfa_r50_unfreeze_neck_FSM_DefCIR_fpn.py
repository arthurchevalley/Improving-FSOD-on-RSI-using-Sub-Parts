_base_ = ['../_base_/models/faster_rcnn_DefCIR_r50_FSM.py']
model = dict(
    type='TFA',
    backbone=dict(frozen_stages=4),
    
    frozen_parameters=[
        'backbone', 
        'neck.fpn_convs',
        #'rpn_head', 
        #'roi_head.bbox_head.shared_fcs', 
        #'roi_head.bbox_head.fc_cls.bias'
    ],
    
    roi_head=dict(
        bbox_head=dict(
            type='CosineSimBBoxHead',
            num_shared_fcs=2,
            num_classes=20,
            cls_bias = True,
            scale=20)))
