_base_ = ['../_base_/models/CRFCNN_r50_cosine_more.py']
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
        )
            
            
    )
