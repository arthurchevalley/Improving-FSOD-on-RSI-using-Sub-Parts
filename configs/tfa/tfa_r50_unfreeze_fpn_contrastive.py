_base_ = ['../_base_/models/CRFCNN_r50.py']
model = dict(

    backbone=dict(frozen_stages=4),
    
    frozen_parameters=[
        'backbone', 
        'neck',
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
            scale=20)
        ),
    c_roi_head=dict(
        bbox_head=dict(
            loss_cls=dict(
                loss_weight=.5),
            )
        )
            
            
    )
